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

///////////////////////////////////Pau//////////////////////////////////

using namespace SCN;

//some globals
GFX::Mesh sphere;
GFX::Mesh cube;

GFX::FBO* gbuffers = nullptr;


Renderer::Renderer(const char* shader_atlas_filename)
{
	render_wireframe = false;
	render_boundaries = false;
	deactivate_ambient_light = false;
	albedo_texture = false;
	emissive_texture = false;
	occlusion_texture = false;
	metallicRoughness_texture = false;
	normalMap_texture = false;
	white_textures = false;
	skip_lights = false;
	skip_shadows = false;
	scene = nullptr;
	skybox_cubemap = nullptr;
	moon_light = nullptr;

	pipeline_mode = ePipelineMode::DEFERRED;
	show_gbuffer = eShowGBuffer::NONE;

	shadow_map_size = 1024;

	if (!GFX::Shader::LoadAtlas(shader_atlas_filename))
		exit(1);
	GFX::checkGLErrors();

	sphere.createSphere(1.0f);
	sphere.uploadToVRAM();

	for (int i = 0; i < NUM_SHADOW_MAPS; i++) {
		shadow_maps[i] = nullptr;
	}
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

		Renderable re;
		re.model = node_model;
		re.mesh = node->mesh;
		re.material = node->material;
		re.distance_to_camera = camera->eye.distance(world_bounding.center);
		re.bounding = world_bounding;
		renderables.push_back(re);
		if (re.material->alpha_mode == SCN::eAlphaMode::BLEND) {
			alphaRenderables.push_back(re);
		}
		else {
			opaqueRenderables.push_back(re);
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
	opaqueRenderables.clear();
	alphaRenderables.clear();
	moon_light = nullptr;

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
			if (!moon_light && light->light_type == SCN::eLightType::DIRECTIONAL) {
				moon_light = light;
			}
		}
	}
}

void Renderer::generateShadowMaps(Camera* main_camera) {
	shadow_map_index = 0;
	
	for (LightEntity* light : lights) {
		light->has_shadow_map = false;;
		if (light->light_type == SCN::eLightType::POINT || !light->cast_shadows) {
			continue;
		}
		else {
			light->has_shadow_map = true;
			if (shadow_maps[shadow_map_index] == nullptr || shadow_maps[shadow_map_index]->width != shadow_map_size) {
				if (shadow_maps[shadow_map_index])
					delete shadow_maps[shadow_map_index];
				shadow_maps[shadow_map_index] = new GFX::FBO();
				shadow_maps[shadow_map_index]->setDepthOnly(shadow_map_size, shadow_map_size);
			}
			shadow_maps[shadow_map_index]->bind();
			glClear(GL_DEPTH_BUFFER_BIT);

			Camera camera;
			vec3 pos= light->getGlobalPosition();
			if (light->light_type == SCN::eLightType::DIRECTIONAL) {
				pos = main_camera->eye;
				camera.lookAt(pos, pos + light->root.global_model.frontVector() * -1.0f, vec3(0, 1, 0));
				camera.setOrthographic(light->area * -0.5, light->area * 0.5, light->area * -0.5, light->area * 0.5, light->near_distance, light->max_distance);
			}
			else if (light->light_type == SCN::eLightType::SPOT) {
				vec3 front = light->root.model.rotateVector(vec3(0, 0, 1));
				camera.lookAt(pos, pos + front * -1.0f, vec3(0, 1, 0));
				camera.setPerspective(light->cone_info.y, 1.0f, light->near_distance, light->max_distance);
			}

			// compute texel size in world units, where frustum size is the distance from left to right in the camera
			float grid = light->area / (float)shadow_maps[shadow_map_index]->width;

			camera.enable();

			//snap camera X,Y to that size in camera space assumingthe frustum is square, otherwise compute gridx and gridy
			camera.view_matrix.M[3][0] = round(camera.view_matrix.M[3][0] / grid) * grid;
			camera.view_matrix.M[3][1] = round(camera.view_matrix.M[3][1] / grid) * grid;

			//update viewproj matrix (be sure no one changes it)
			camera.viewprojection_matrix = camera.view_matrix * camera.projection_matrix;

			for (auto& re : opaqueRenderables) {
				if (camera.testBoxInFrustum(re.bounding.center, re.bounding.halfsize))
					renderMeshWithMaterialFlat(re.model, re.mesh, re.material);
			}
			light->shadowmap_view_projection = camera.viewprojection_matrix;

			shadow_maps[shadow_map_index]->unbind();

			if (shadow_map_index < NUM_SHADOW_MAPS - 1)
				shadow_map_index++;
			else
				printf("Max number of shadow maps achieved, incrise it to be able to have more shadow maps (NUM_SHADOW_MAPS)");
		}
	}
}
////////////////


void Renderer::renderScene(SCN::Scene* scene, Camera* camera)
{
	this->scene = scene;
	setupScene();
	extractSceneInfo(scene, camera);
	if (!skip_shadows)
		generateShadowMaps(camera);

	if (pipeline_mode == ePipelineMode::FORWARD)
		renderSceneForward(scene, camera);
	else if (pipeline_mode == ePipelineMode::DEFERRED)
		renderSceneDeferred(scene, camera);


	opaqueRenderables.clear();
	alphaRenderables.clear();
}

void Renderer::renderSceneForward(SCN::Scene* scene, Camera* camera)
{
	camera->enable();

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
	// Sort by distance_to_camera from far to near
	std::sort(alphaRenderables.begin(), alphaRenderables.end(), [](Renderable& a, Renderable& b) {return (a.distance_to_camera > b.distance_to_camera);});

	// Render opaque objects first
	for (Renderable& re : opaqueRenderables) {
		//if bounding box inside camera
		if (camera->testBoxInFrustum(re.bounding.center, re.bounding.halfsize))
			if (skip_lights) {
				renderMeshWithMaterial(re.model, re.mesh, re.material);
			}
			else {
				renderMeshWithMaterialLights(re.model, re.mesh, re.material);
			}
	}

	// Then render alpha objects
	for (Renderable& re : alphaRenderables) {
		//if bounding box inside camera
		if (camera->testBoxInFrustum(re.bounding.center, re.bounding.halfsize))
			if (skip_lights) {
				renderMeshWithMaterial(re.model, re.mesh, re.material);
			}
			else {
				renderMeshWithMaterialLights(re.model, re.mesh, re.material);
			}
	}
	////////////////
}

void Renderer::renderSceneDeferred(SCN::Scene* scene, Camera* camera) {

	vec2 size = CORE::getWindowSize();

	//generear los GBuffers
	if (!gbuffers)
	{
		gbuffers = new GFX::FBO();
		gbuffers->create(size.x, size.y, 4, GL_RGBA, GL_UNSIGNED_BYTE, true);
	}
	gbuffers->bind();
	camera->enable();

	glEnable(GL_DEPTH_TEST);

	glClearColor(0, 0, 0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Sort by distance_to_camera from near to far to avoid overdraw
	std::sort(opaqueRenderables.begin(), opaqueRenderables.end(), [](Renderable& a, Renderable& b) {return (a.distance_to_camera < b.distance_to_camera); });

	// Render opaque objects first
	for (Renderable& re : opaqueRenderables) {
		//if bounding box inside camera
		if (camera->testBoxInFrustum(re.bounding.center, re.bounding.halfsize))
			renderMeshWithMaterialGBuffers(re.model, re.mesh, re.material);
	}

	gbuffers->unbind();

	glClearColor(scene->background_color.x, scene->background_color.y, scene->background_color.z, 1.0);
	glClearColor(0, 0, 0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	GFX::checkGLErrors();

	if (skybox_cubemap)
		renderSkybox(skybox_cubemap);

	//draw the lights
	GFX::Mesh* quad = GFX::Mesh::getQuad();
	GFX::Shader* deferred_global = GFX::Shader::Get("deferred_global");
	assert(deferred_global);
	deferred_global->enable();
	deferred_global->setUniform("u_color_texture", gbuffers->color_textures[0], 0);
	deferred_global->setUniform("u_normal_texture", gbuffers->color_textures[1], 1);
	deferred_global->setUniform("u_emissive_occlusion_texture", gbuffers->color_textures[2], 2);
	deferred_global->setUniform("u_normalmap_texture", gbuffers->color_textures[3], 3);
	deferred_global->setUniform("u_depth_texture", gbuffers->depth_texture, 4);

	deferred_global->setUniform("u_ambient_light", scene->ambient_light);
	deferred_global->setUniform("u_iRes", vec2(1.0 / size.x, 1.0 / size.y));
	deferred_global->setUniform("u_inverse_viewprojection", camera->inverse_viewprojection_matrix);
	deferred_global->setUniform("u_emissive_first", vec3(1.0));
	deferred_global->setUniform("u_norm_contr", normalMap_texture);

	shadow_map_index = 0;
	if (lights.size() && (!skip_lights)) {

		glEnable(GL_DEPTH_TEST);

		glDepthFunc(GL_ALWAYS);
		glDisable(GL_BLEND);

		for (LightEntity* light : lights) {
			lightToShader(light, deferred_global);

			if (light->has_shadow_map) {
				deferred_global->setUniform("u_light_cast_shadow", !skip_shadows ? 1 : 0);
				deferred_global->setUniform("u_shadow_map", shadow_maps[shadow_map_index]->depth_texture, 8);
				deferred_global->setUniform("u_shadow_map_view_projection", light->shadowmap_view_projection);
				deferred_global->setUniform("u_shadow_bias", light->shadow_bias);
				if (shadow_map_index < NUM_SHADOW_MAPS - 1)
					shadow_map_index++;
				else
					printf("Max number of shadow maps achieved, incrise it to be able to have more shadow maps (NUM_SHADOW_MAPS)");
			}

			quad->render(GL_TRIANGLES);
			deferred_global->setUniform("u_ambient_light", vec3(0.0));
			deferred_global->setUniform("u_emissive_first", vec3(0.0));
			glEnable(GL_BLEND);
			glBlendFunc(GL_ONE, GL_ONE);
			glDisable(GL_DEPTH_TEST);
		}
	}
	else {
		deferred_global->setUniform("u_light_type", 0);
		quad->render(GL_TRIANGLES);
	}

	glDisable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);

	// Sort by distance_to_camera from far to near
	std::sort(alphaRenderables.begin(), alphaRenderables.end(), [](Renderable& a, Renderable& b) {return (a.distance_to_camera > b.distance_to_camera); });

	// Then render alpha objects
	for (Renderable& re : alphaRenderables) {
		//if bounding box inside camera
		if (camera->testBoxInFrustum(re.bounding.center, re.bounding.halfsize))
			if (skip_lights) {
				renderMeshWithMaterial(re.model, re.mesh, re.material);
			}
			else {
				renderMeshWithMaterialLights(re.model, re.mesh, re.material);
			}
	}


	if (show_gbuffer == eShowGBuffer::COLOR)
		gbuffers->color_textures[0]->toViewport();
	if (show_gbuffer == eShowGBuffer::NORMAL)
		gbuffers->color_textures[1]->toViewport();
	if (show_gbuffer == eShowGBuffer::EMISSIVE)
		gbuffers->color_textures[2]->toViewport();
	/*if (show_gbuffer == eShowGBuffer::OCCLUSION)
		gbuffers->color_textures[2]->toViewport();*/
	if (show_gbuffer == eShowGBuffer::NORMALMAP)
		gbuffers->color_textures[3]->toViewport();
	if (show_gbuffer == eShowGBuffer::DEPTH)  //Needs to be done use the depth shader
	{
		GFX::Shader* depth_shader = GFX::Shader::Get("depth");
		depth_shader->enable();
		depth_shader->setUniform("u_camera_nearfar", vec2(camera->near_plane, camera->far_plane));
		gbuffers->depth_texture->toViewport(depth_shader);
	}
	if (show_gbuffer == eShowGBuffer::GBUFFERS) {
		//set an area of the screen and render fullscreen quad
		glViewport(0, size.y * 0.5, size.x * 0.5, size.y * 0.5);
		gbuffers->color_textures[0]->toViewport(); //colorbuffer

		glViewport(size.x * 0.5, size.y * 0.5, size.x * 0.5, size.y * 0.5);
		gbuffers->color_textures[3]->toViewport(); //normalbuffer

		glViewport(size.x * 0.5, 0.0, size.x * 0.5, size.y * 0.5);
		gbuffers->color_textures[2]->toViewport(); //emissivelbuffer

		//for the depth remember to linearize when displaying it
		glViewport(0, 0, size.x * 0.5, size.y * 0.5);
		GFX::Shader* depth_shader = GFX::Shader::Get("depth");
		depth_shader->enable();
		vec2 near_far = vec2(camera->near_plane, camera->far_plane);
		depth_shader->setUniform("u_camera_nearfar", near_far);
		gbuffers->depth_texture->toViewport(depth_shader);

		//set the viewport back to full screen
		glViewport(0, 0, size.x, size.y);

	}
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
void Renderer::renderMeshWithMaterialFlat(const Matrix44 model, GFX::Mesh* mesh, SCN::Material* material)
{
	//in case there is nothing to do
	if (!mesh || !mesh->getNumVertices() || !material)
		return;
	assert(glGetError() == GL_NO_ERROR);

	//define locals to simplify coding
	GFX::Shader* shader = NULL;
	GFX::Texture* texture = NULL;
	Camera* camera = Camera::current;

	texture = material->textures[SCN::eTextureChannel::ALBEDO].texture;

	if (texture == NULL)
		texture = GFX::Texture::getWhiteTexture(); //a 1x1 white texture

	glDisable(GL_BLEND);

	//select if render both sides of the triangles
	if (material->two_sided)
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

	shader->setUniform("u_color", material->color);
	if (texture)
		shader->setUniform("u_texture", texture, 0);

	//this is used to say which is the alpha threshold to what we should not paint a pixel on the screen (to cut polygons according to texture alpha)
	shader->setUniform("u_alpha_cutoff", material->alpha_mode == SCN::eAlphaMode::MASK ? material->alpha_cutoff : 0.001f);

	//do the draw call that renders the mesh into the screen
	mesh->render(GL_TRIANGLES);

	//disable shader
	shader->disable();
}

//renders a mesh given its transform and material
void Renderer::renderMeshWithMaterialGBuffers(const Matrix44 model, GFX::Mesh* mesh, SCN::Material* material)
{
	//in case there is nothing to do
	if (!mesh || !mesh->getNumVertices() || !material)
		return;
	assert(glGetError() == GL_NO_ERROR);

	//define locals to simplify coding
	GFX::Shader* shader = NULL;
	GFX::Texture* texture = NULL;
	GFX::Texture* textureEmissive = NULL;
	GFX::Texture* textureMetallicRoughness = NULL;
	GFX::Texture* textureNormalMap = NULL;
	GFX::Texture* textureOcclusion = NULL;
	Camera* camera = Camera::current;

	if (!white_textures) {
		if (!albedo_texture)
			texture = material->textures[SCN::eTextureChannel::ALBEDO].texture;
		if (!emissive_texture)
			textureEmissive = material->textures[SCN::eTextureChannel::EMISSIVE].texture;
		if (!metallicRoughness_texture)
			textureMetallicRoughness = material->textures[SCN::eTextureChannel::METALLIC_ROUGHNESS].texture;
		if (!normalMap_texture)
			textureNormalMap = material->textures[SCN::eTextureChannel::NORMALMAP].texture;
		if (!occlusion_texture)
			textureOcclusion = material->textures[SCN::eTextureChannel::OCCLUSION].texture;
	}

	if (texture == NULL)
		texture = GFX::Texture::getWhiteTexture(); //a 1x1 white texture
	if (textureEmissive == NULL)
		textureEmissive = GFX::Texture::getWhiteTexture(); //a 1x1 white 
	if (textureMetallicRoughness == NULL)
		textureMetallicRoughness = GFX::Texture::getWhiteTexture(); //a 1x1 white 
	if (textureNormalMap == NULL)
		textureNormalMap = GFX::Texture::getWhiteTexture(); //a 1x1 white 
	if (textureOcclusion == NULL)
		textureOcclusion = GFX::Texture::getWhiteTexture(); //a 1x1 white texture

	glDisable(GL_BLEND);

	//select if render both sides of the triangles
	if (material->two_sided)
		glDisable(GL_CULL_FACE);
	else
		glEnable(GL_CULL_FACE);
	assert(glGetError() == GL_NO_ERROR);

	glEnable(GL_DEPTH_TEST);

	//chose a shader
	shader = GFX::Shader::Get("gbuffers");
	assert(shader);
	assert(glGetError() == GL_NO_ERROR);

	//no shader? then nothing to render
	if (!shader)
		return;
	shader->enable();

	//upload uniforms
	shader->setUniform("u_model", model);
	cameraToShader(camera, shader);

	shader->setUniform("u_color", material->color);
	shader->setUniform("u_texture", texture, 0);
	shader->setUniform("u_texture_emissive", textureEmissive, 1);
	if (!white_textures && !emissive_texture) {
		shader->setUniform("u_emissive_factor", material->emissive_factor);
	}
	else {
		shader->setUniform("u_emissive_factor", vec3(0.0));
	}
	shader->setUniform("u_texture_metallic_roughness", textureMetallicRoughness, 2);
	shader->setUniform("u_texture_normalmap", textureNormalMap, 3);
	shader->setUniform("u_texture_occlusion", textureOcclusion, 4);

	//this is used to say which is the alpha threshold to what we should not paint a pixel on the screen (to cut polygons according to texture alpha)
	shader->setUniform("u_alpha_cutoff", material->alpha_mode == SCN::eAlphaMode::MASK ? material->alpha_cutoff : 0.001f);

	//do the draw call that renders the mesh into the screen
	mesh->render(GL_TRIANGLES);

	//disable shader
	shader->disable();
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
	
	if (!white_textures) {
		texture = material->textures[SCN::eTextureChannel::ALBEDO].texture;
	}
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
	GFX::Texture* textureMetallicRoughness = NULL;
	GFX::Texture* textureNormalMap = NULL;
	GFX::Texture* textureOcclusion = NULL;
	Camera* camera = Camera::current;
	if (!white_textures) {
		if(!albedo_texture)
			textureAlbedo = material->textures[SCN::eTextureChannel::ALBEDO].texture;
		if (!emissive_texture)
			textureEmissive = material->textures[SCN::eTextureChannel::EMISSIVE].texture;
		if (!metallicRoughness_texture)
			textureMetallicRoughness = material->textures[SCN::eTextureChannel::METALLIC_ROUGHNESS].texture;
		if (!normalMap_texture)
			textureNormalMap = material->textures[SCN::eTextureChannel::NORMALMAP].texture;
		if (!occlusion_texture)
			textureOcclusion = material->textures[SCN::eTextureChannel::OCCLUSION].texture;
	}

	if (textureAlbedo == NULL)
		textureAlbedo = GFX::Texture::getWhiteTexture(); //a 1x1 white texture
	if (textureEmissive == NULL)
		textureEmissive = GFX::Texture::getWhiteTexture(); //a 1x1 white 
	if (textureMetallicRoughness == NULL)
		textureMetallicRoughness = GFX::Texture::getWhiteTexture(); //a 1x1 white 
	if (textureNormalMap == NULL)
		textureNormalMap = GFX::Texture::getWhiteTexture(); //a 1x1 white 
	if (textureOcclusion == NULL)
		textureOcclusion = GFX::Texture::getWhiteTexture(); //a 1x1 white texture

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
	if (!deactivate_ambient_light) {
		shader->setUniform("u_ambient_light", scene->ambient_light);
	}
	else {
		shader->setUniform("u_ambient_light", vec3(0.0));
	}
	if (!white_textures && !emissive_texture) {
		shader->setUniform("u_emissive_factor", material->emissive_factor);
	}
	else {
		shader->setUniform("u_emissive_factor", vec3(0.0));
	}
	shader->setUniform("u_norm_contr", normalMap_texture);

	shader->setUniform("u_color", material->color);
	shader->setUniform("u_texture_albedo", textureAlbedo, 0);
	shader->setUniform("u_texture_emissive", textureEmissive, 1);
	shader->setUniform("u_texture_metallic_roughness", textureMetallicRoughness, 2);
	shader->setUniform("u_texture_normalmap", textureNormalMap, 3);
	shader->setUniform("u_texture_occlusion", textureOcclusion, 4);
	///////////////
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
	int shadow_map_index = 0;
	if (lights.size()) {
		for (LightEntity* light : lights) {
			lightToShader(light, shader);

			if (light->has_shadow_map) {
				shader->setUniform("u_light_cast_shadow", !skip_shadows ? 1 : 0);
				shader->setUniform("u_shadow_map", shadow_maps[shadow_map_index]->depth_texture, 8);
				shader->setUniform("u_shadow_map_view_projection", light->shadowmap_view_projection);
				shader->setUniform("u_shadow_bias", light->shadow_bias);
				if (shadow_map_index < NUM_SHADOW_MAPS - 1)
					shadow_map_index++;
				else
					printf("Max number of shadow maps achieved, incrise it to be able to have more shadow maps (NUM_SHADOW_MAPS)");
			}

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
	
	ImGui::Combo("Pipeline", (int*)&pipeline_mode, "FLAT\0FORWARD\0DEFERRED\0", ePipelineMode::PIPELINE_COUNT);
	ImGui::Combo("GBuffers", (int*)&show_gbuffer, "NONE\0COLOR\0NORMALMAP\0NORMAL\0DEPTH\0EMISSIVE\0OCCLUSION\0GBUFFERS", eShowGBuffer::GBUFFERS_COUNT);

	ImGui::Checkbox("Wireframe", &render_wireframe);
	ImGui::Checkbox("Boundaries", &render_boundaries);

	ImGui::Checkbox("Deactivate_ambient_light", &deactivate_ambient_light);
	ImGui::Checkbox("Deactivate_Albedo_texture", &albedo_texture);
	ImGui::Checkbox("Deactivate_Emissive_texture", &emissive_texture);
	ImGui::Checkbox("Deactivate_MetallicRoughness_texture", &metallicRoughness_texture);
	ImGui::Checkbox("Deactivate_NormalMap_texture", &normalMap_texture);
	ImGui::Checkbox("Deactivate_Occlusion_texture", &occlusion_texture);
	ImGui::Checkbox("Remove_textures", &white_textures);
	ImGui::Checkbox("Remove_lights", &skip_lights);
	ImGui::Checkbox("Remove_shadows", &skip_shadows);


	// Create a slider for the exponent
	if (ImGui::SliderInt("Shadowmap Size", &power_of_two, 7, 12)) {
		// Calculate the actual shadowmap size as a power of two
		shadow_map_size = (1 << power_of_two);
	}
	// Display the actual shadowmap size
	ImGui::Text("Actual Shadowmap Size: %d", shadow_map_size);
}

#else
void Renderer::showUI() {}
#endif
