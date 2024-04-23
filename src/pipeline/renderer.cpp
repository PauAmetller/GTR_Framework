#include "renderer.h"

#include <algorithm> //sort

#include "camera.h"
#include "../gfx/gfx.h"
#include "../gfx/shader.h"
#include "../gfx/mesh.h"
#include "../gfx/texture.h"
#include "../gfx/fbo.h"
#include "../pipeline/prefab.h"
#include "../pipeline/material.h"
#include "../pipeline/animation.h"
#include "../utils/utils.h"
#include "../extra/hdre.h"
#include "../core/ui.h"

#include "scene.h"


using namespace SCN;

//some globals
GFX::Mesh sphere;

Renderer::Renderer(const char* shader_atlas_filename)
{
	render_wireframe = false;
	render_boundaries = false;
	scene = nullptr;
	skybox_cubemap = nullptr;

	if (!GFX::Shader::LoadAtlas(shader_atlas_filename))
		exit(1);
	GFX::checkGLErrors();

	sphere.createSphere(1.0f);
	sphere.uploadToVRAM();
}

void Renderer::setupScene()
{
	if (scene->skybox_filename.size())
		skybox_cubemap = GFX::Texture::Get(std::string(scene->base_folder + "/" + scene->skybox_filename).c_str());
	else
		skybox_cubemap = nullptr;
}

//////////////////
void Renderer::extractRenderables(SCN::Node* node, Camera* camera) {

	if (!node->visible)
		return;

	Matrix44 node_model = node->getGlobalMatrix(true);

	//Render if node has mesh and material
	if (node->material && node->mesh) {

		//compute the bounding box of the object in the world space
		BoundingBox world_bounding = transformBoundingBox(node_model, node->mesh->box);

		//if bounding box inside camera
		if (camera->testBoxInFrustum(world_bounding.center, world_bounding.halfsize)) {
			Renderable re;
			re.model = node_model;
			re.mesh = node->mesh;
			re.material = node->material;
			re.distance_to_camera = camera->eye.distance(world_bounding.center);
			re.bounding = world_bounding;
			renderables.push_back(re);
		}
	}

	//iterate recursibely with its children entities
	for (size_t i = 0; i < node->children.size(); i++) {
		extractRenderables(node->children[i], camera);
	}
}

void Renderer::extractSceneInfo(SCN::Scene* scene, Camera* camera) {

	renderables.clear();
	lights.clear();

	//prepare entities
	for (size_t i = 0; i < scene->entities.size(); i++) {

		BaseEntity* ent = scene->entities[i];

		//Skip if entity not visible
		if (!ent->visible)
			continue;

		if(ent->getType() == eEntityType::PREFAB){

			PrefabEntity* pent = (SCN::PrefabEntity*)ent;
			if (pent->prefab) {
				extractRenderables(&pent->root, camera);
			}
		} else if (ent->getType() == eEntityType::LIGHT) {

			LightEntity* light = (SCN::LightEntity*)ent;
			mat4 model = light->root.getGlobalMatrix();

			if (light->light_type == SCN::eLightType::DIRECTIONAL || camera->testSphereInFrustum(model.getTranslation(), light->max_distance)) {
				lights.push_back(light);
			}
		}
	}
}

////////////////


void Renderer::renderScene(SCN::Scene* scene, Camera* camera)
{
	this->scene = scene;
	setupScene();
	extractSceneInfo(scene, camera);

	glDisable(GL_BLEND);
	glEnable(GL_DEPTH_TEST);

	//set the clear color (the background color)
	glClearColor(scene->background_color.x, scene->background_color.y, scene->background_color.z, 1.0);

	// Clear the color and the depth buffer
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	GFX::checkGLErrors();

	//render skybox
	if(skybox_cubemap)
		renderSkybox(skybox_cubemap);

	//////////////
	std::vector<Renderable> opaqueRenderables;
	std::vector<Renderable> alphaRenderables;

	//Separates the opaque and alpha renderables into two lists
	for (Renderable& re : renderables) {
		if (re.material->alpha_mode == SCN::eAlphaMode::NO_ALPHA) { 
			opaqueRenderables.push_back(re);
		}
		else {
			alphaRenderables.push_back(re);
		}
	}

	// Sort by distance_to_camera from far to near
	std::sort(alphaRenderables.begin(), alphaRenderables.end(), [](Renderable& a, Renderable& b) {return (a.distance_to_camera > b.distance_to_camera);});

	// Render opaque objects first
	for (Renderable& re : opaqueRenderables) {
		renderMeshWithMaterialLights(re.model, re.mesh, re.material);
	}

	// Then render alpha objects
	for (Renderable& re : alphaRenderables) {
		renderMeshWithMaterialLights(re.model, re.mesh, re.material);
	}
	opaqueRenderables.clear();
	alphaRenderables.clear();
	////////////////
}


void Renderer::renderSkybox(GFX::Texture* cubemap)
{
	Camera* camera = Camera::current;

	glDisable(GL_BLEND);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	if (render_wireframe)
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	GFX::Shader* shader = GFX::Shader::Get("skybox");
	if (!shader)
		return;
	shader->enable();

	Matrix44 m;
	m.setTranslation(camera->eye.x, camera->eye.y, camera->eye.z);
	m.scale(10, 10, 10);
	shader->setUniform("u_model", m);
	cameraToShader(camera, shader);
	shader->setUniform("u_texture", cubemap, 0);
	sphere.render(GL_TRIANGLES);
	shader->disable();
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glEnable(GL_DEPTH_TEST);
}

//renders a node of the prefab and its children
void Renderer::renderNode(SCN::Node* node, Camera* camera)
{
	if (!node->visible)
		return;

	//compute global matrix
	Matrix44 node_model = node->getGlobalMatrix(true);

	//does this node have a mesh? then we must render it
	if (node->mesh && node->material)
	{
		//compute the bounding box of the object in world space (by using the mesh bounding box transformed to world space)
		BoundingBox world_bounding = transformBoundingBox(node_model,node->mesh->box);
		
		//if bounding box is inside the camera frustum then the object is probably visible
		if (camera->testBoxInFrustum(world_bounding.center, world_bounding.halfsize) )
		{
			if(render_boundaries)
				node->mesh->renderBounding(node_model, true);
			renderMeshWithMaterial(node_model, node->mesh, node->material);
		}
	}

	//iterate recursively with children
	for (int i = 0; i < node->children.size(); ++i)
		renderNode( node->children[i], camera);
}

//renders a mesh given its transform and material
void Renderer::renderMeshWithMaterial(const Matrix44 model, GFX::Mesh* mesh, SCN::Material* material)
{
	//in case there is nothing to do
	if (!mesh || !mesh->getNumVertices() || !material )
		return;
    assert(glGetError() == GL_NO_ERROR);

	//define locals to simplify coding
	GFX::Shader* shader = NULL;
	GFX::Texture* texture = NULL;
	Camera* camera = Camera::current;
	
	texture = material->textures[SCN::eTextureChannel::ALBEDO].texture;
	//texture = material->emissive_texture;
	//texture = material->metallic_roughness_texture;
	//texture = material->normal_texture;
	//texture = material->occlusion_texture;
	if (texture == NULL)
		texture = GFX::Texture::getWhiteTexture(); //a 1x1 white texture

	//select the blending
	if (material->alpha_mode == SCN::eAlphaMode::BLEND)
	{
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	}
	else
		glDisable(GL_BLEND);

	//select if render both sides of the triangles
	if(material->two_sided)
		glDisable(GL_CULL_FACE);
	else
		glEnable(GL_CULL_FACE);
    assert(glGetError() == GL_NO_ERROR);

	glEnable(GL_DEPTH_TEST);

	//chose a shader
	shader = GFX::Shader::Get("texture");

    assert(glGetError() == GL_NO_ERROR);

	//no shader? then nothing to render
	if (!shader)
		return;
	shader->enable();

	//upload uniforms
	shader->setUniform("u_model", model);
	cameraToShader(camera, shader);
	float t = getTime();
	shader->setUniform("u_time", t );

	shader->setUniform("u_color", material->color);
	if(texture)
		shader->setUniform("u_texture", texture, 0);

	//this is used to say which is the alpha threshold to what we should not paint a pixel on the screen (to cut polygons according to texture alpha)
	shader->setUniform("u_alpha_cutoff", material->alpha_mode == SCN::eAlphaMode::MASK ? material->alpha_cutoff : 0.001f);

	if (render_wireframe)
		glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );

	//do the draw call that renders the mesh into the screen
	mesh->render(GL_TRIANGLES);

	//disable shader
	shader->disable();

	//set the render state as it was before to avoid problems with future renders
	glDisable(GL_BLEND);
	glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
}

//renders a mesh given its transform and material, with lights /////////////
void Renderer::renderMeshWithMaterialLights(const Matrix44 model, GFX::Mesh* mesh, SCN::Material* material)
{
	//in case there is nothing to do
	if (!mesh || !mesh->getNumVertices() || !material)
		return;
	assert(glGetError() == GL_NO_ERROR);

	//define locals to simplify coding
	GFX::Shader* shader = NULL;
	GFX::Texture* textureAlbedo = NULL;
	GFX::Texture* textureEmissive = NULL;
	Camera* camera = Camera::current;

	textureAlbedo = material->textures[SCN::eTextureChannel::ALBEDO].texture;
	textureEmissive = material->textures[SCN::eTextureChannel::EMISSIVE].texture;
	//texture = material->emissive_texture;
	//texture = material->metallic_roughness_texture;
	//texture = material->normal_texture;
	//texture = material->occlusion_texture;
	if (textureAlbedo == NULL)
		textureAlbedo = GFX::Texture::getWhiteTexture(); //a 1x1 white texture
	if (textureEmissive == NULL)
		textureEmissive = GFX::Texture::getWhiteTexture(); //a 1x1 white texture

	//select the blending
	if (material->alpha_mode == SCN::eAlphaMode::BLEND)
	{
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	}
	else
		glDisable(GL_BLEND);

	//select if render both sides of the triangles
	if (material->two_sided)
		glDisable(GL_CULL_FACE);
	else
		glEnable(GL_CULL_FACE);
	assert(glGetError() == GL_NO_ERROR);

	glEnable(GL_DEPTH_TEST);

	//chose a shader
	shader = GFX::Shader::Get("light");

	assert(glGetError() == GL_NO_ERROR);

	//no shader? then nothing to render
	if (!shader)
		return;
	shader->enable();

	//upload uniforms
	shader->setUniform("u_model", model);
	cameraToShader(camera, shader);
	float t = getTime();
	shader->setUniform("u_time", t);
	////////////
	shader->setUniform("u_ambient_light", scene->ambient_light);
	shader->setUniform("u_emissive_factor", material->emissive_factor);
	///////////
	shader->setUniform("u_color", material->color);
	shader->setUniform("u_texture_albedo", textureAlbedo, 0);
	shader->setUniform("u_texture_emissive", textureEmissive, 1);

	//this is used to say which is the alpha threshold to what we should not paint a pixel on the screen (to cut polygons according to texture alpha)
	shader->setUniform("u_alpha_cutoff", material->alpha_mode == SCN::eAlphaMode::MASK ? material->alpha_cutoff : 0.001f);

	if (render_wireframe)
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	////////////
	if (material->alpha_mode != SCN::eAlphaMode::BLEND){
		glDisable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE);
	}
	glDepthFunc(GL_LEQUAL);

	//multi_pass
	if (lights.size()) {
		for (LightEntity* light : lights) {
			lightToShader(light, shader);
			mesh->render(GL_TRIANGLES);
			glEnable(GL_BLEND);
			shader->setUniform("u_ambient_light", vec3(0.0));
			shader->setUniform("u_emissive_factor", vec3(0.0));
		}
	}
	else {
		shader->setUniform("u_light_type", 0);
		mesh->render(GL_TRIANGLES);
	}

	glDepthFunc(GL_LESS);
	//////////////
	
	//disable shader
	shader->disable();

	//set the render state as it was before to avoid problems with future renders
	glDisable(GL_BLEND);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}
/////////////

void SCN::Renderer::cameraToShader(Camera* camera, GFX::Shader* shader)
{
	shader->setUniform("u_viewprojection", camera->viewprojection_matrix );
	shader->setUniform("u_camera_position", camera->eye);
}

void SCN::Renderer::lightToShader(LightEntity* light, GFX::Shader* shader) 
{
	shader->setUniform("u_light_type", (int)light->light_type);
	shader->setUniform("u_light_position", light->root.global_model.getTranslation());
	shader->setUniform("u_light_color", light->color * light->intensity);
	shader->setUniform("u_light_max_distance", light->max_distance);
	shader->setUniform("u_light_cone_info", vec2(cos(light->cone_info.x * DEG2RAD), cos(light->cone_info.y * DEG2RAD)));
	shader->setUniform("u_light_front", light->root.model.frontVector().normalize());
}

#ifndef SKIP_IMGUI

void Renderer::showUI()
{
		
	ImGui::Checkbox("Wireframe", &render_wireframe);
	ImGui::Checkbox("Boundaries", &render_boundaries);

	//add here your stuff
	//...
}

#else
void Renderer::showUI() {}
#endif