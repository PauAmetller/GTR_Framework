//example of some shaders compiled
flat basic.vs flat.fs
texture basic.vs texture.fs
light basic.vs light.fs
skybox basic.vs skybox.fs
depth quad.vs depth.fs
multi basic.vs multi.fs
gbuffers basic.vs gbuffers.fs
deferred_global quad.vs deferred_global.fs
deferred_ws basic.vs deferred_global.fs
ssao quad.vs ssao.fs

\basic.vs

#version 330 core

in vec3 a_vertex;
in vec3 a_normal;
in vec2 a_coord;
in vec4 a_color;

uniform vec3 u_camera_pos;

uniform mat4 u_model;
uniform mat4 u_viewprojection;

//this will store the color for the pixel shader
out vec3 v_position;
out vec3 v_world_position;
out vec3 v_normal;
out vec2 v_uv;
out vec4 v_color;

uniform float u_time;

void main()
{	
	//calcule the normal in camera space (the NormalMatrix is like ViewMatrix but without traslation)
	v_normal = (u_model * vec4( a_normal, 0.0) ).xyz;
	
	//calcule the vertex in object space
	v_position = a_vertex;
	v_world_position = (u_model * vec4( v_position, 1.0) ).xyz;
	
	//store the color in the varying var to use it from the pixel shader
	v_color = a_color;

	//store the texture coordinates
	v_uv = a_coord;

	//calcule the position of the vertex using the matrices
	gl_Position = u_viewprojection * vec4( v_world_position, 1.0 );
}

\quad.vs

#version 330 core

in vec3 a_vertex;
in vec2 a_coord;
out vec2 v_uv;

void main()
{	
	v_uv = a_coord;
	gl_Position = vec4( a_vertex, 1.0 );
}


\flat.fs

#version 330 core

uniform vec4 u_color;

out vec4 FragColor;

void main()
{
	FragColor = u_color;
}

\ComputeShadow

//Shadow_map resources
uniform int u_light_cast_shadow;
uniform sampler2D u_shadow_map;
uniform mat4 u_shadow_map_view_projection;
uniform float u_shadow_bias;

float computeShadow( vec3 wp){
	//project our 3D position to the shadowmap
	vec4 proj_pos = u_shadow_map_view_projection * vec4(wp,1.0);

	//from homogeneus space to clip space
	vec2 shadow_uv = proj_pos.xy / proj_pos.w;

	//from clip space to uv space
	shadow_uv = shadow_uv * 0.5 + vec2(0.5);

	//it is outside on the sides, or it is before near or behind far plane
	if( shadow_uv.x < 0.0 || shadow_uv.y < 0.0 || shadow_uv.x > 1.0 || shadow_uv.y > 1.0){
		return 1.0;
	}

	//get point depth [-1 .. +1] in non-linear space
	float real_depth = (proj_pos.z - u_shadow_bias) / proj_pos.w;

	//normalize from [-1..+1] to [0..+1] still non-linear
	real_depth = real_depth * 0.5 + 0.5;

	//it is before near or behind far plane
	if(real_depth < 0.0 || real_depth > 1.0)
		return 1.0;


	//read depth from depth buffer in [0..+1] non-linear
	float shadow_depth = texture( u_shadow_map, shadow_uv).x;

	//compute final shadow factor by comparing
	float shadow_factor = 1.0;

	//we can compare them, even if they are not linear
	if( shadow_depth < real_depth )
		shadow_factor = 0.0;
	return shadow_factor;

}

\specullar_function

float GGX(float NdotV, float k){
	return NdotV / (NdotV * (1.0 - k) + k);
}
	
float G_Smith( float NdotV, float NdotL, float roughness)
{
	float k = pow(roughness + 1.0, 2.0) / 8.0;
	return GGX(NdotL, k) * GGX(NdotV, k);
}

// Fresnel term with scalar optimization(f90=1)
float F_Schlick( const in float VoH, const in float f0)
{
	float f = pow(1.0 - VoH, 5.0);
	return f0 + (1.0 - f0) * f;
}

// Fresnel term with colorized fresnel
vec3 F_Schlick( const in float VoH, const in vec3 f0)
{
	float f = pow(1.0 - VoH, 5.0);
	return f0 + (vec3(1.0) - f0) * f;
}

#define RECIPROCAL_PI 0.3183098861837697
#define PI 3.14159265359

vec3 Fd_Lambert(vec3 color) {
    return color/PI;
}

// Diffuse Reflections: Disney BRDF using retro-reflections using F term, this is much more complex!!
float Fd_Burley ( const in float NoV, const in float NoL,const in float LoH, const in float linearRoughness)
{
        float f90 = 0.5 + 2.0 * linearRoughness * LoH * LoH;
        float lightScatter = F_Schlick(NoL, 1.0);//, f90);  //Check latter
        float viewScatter  = F_Schlick(NoV, 1.0);//, f90);
        return lightScatter * viewScatter * RECIPROCAL_PI;
}

// Normal Distribution Function using GGX Distribution
float D_GGX (	const in float NoH, const in float linearRoughness )
{
	float a2 = linearRoughness * linearRoughness;
	float f = (NoH * NoH) * (a2 - 1.0) + 1.0;
	return a2 / (PI * f * f);
}

//this is the cook torrance specular reflection model
vec3 specularBRDF( float roughness, vec3 f0, float NoH, float NoV, float NoL, float LoH )
{
	float a = roughness * roughness;

	// Normal Distribution Function
	float D = D_GGX( NoH, a );

	// Fresnel Function
	vec3 F = F_Schlick( LoH, f0 );

	// Visibility Function (shadowing/masking)
	float G = G_Smith( NoV, NoL, roughness );
		
	// Norm factor
	vec3 spec = D * G * F;
	spec /= (4.0 * NoL * NoV + 1e-6);

	return spec;
}

\normalmap_functions

mat3 cotangent_frame(vec3 N, vec3 p, vec2 uv)
{
	// get edge vectors of the pixel triangle
	vec3 dp1 = dFdx( p );
	vec3 dp2 = dFdy( p );
	vec2 duv1 = dFdx( uv );
	vec2 duv2 = dFdy( uv );
	
	// solve the linear system
	vec3 dp2perp = cross( dp2, N );
	vec3 dp1perp = cross( N, dp1 );
	vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
	vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;
 
	// construct a scale-invariant frame 
	float invmax = inversesqrt( max( dot(T,T), dot(B,B) ) );
	return mat3( T * invmax, B * invmax, N );
}

vec3 perturbNormal(vec3 N, vec3 WP, vec2 uv, vec3 normal_pixel)
{
	normal_pixel = normal_pixel * 255./127. - 128./127.;
	mat3 TBN = cotangent_frame(N, WP, uv);
	return normalize(TBN * normal_pixel);
}


\ComputeLights

	vec3 light_add;
	float NoL;
	float shadow_factor = 1.0;
	if ( u_light_type == DIRECTIONALLIGHT)
	{
		L = u_light_front;
		NoL = dot(N,L);
		NdotL = clamp( NoL, 0.0, 1.0 );
		light_add = u_light_color;
		if(u_light_cast_shadow == 1)
			shadow_factor = computeShadow(v_world_position);
	}
	else if (u_light_type == SPOTLIGHT || u_light_type == POINTLIGHT) //spot and point
	{
		L = u_light_position - v_world_position;
		float dist = length(L);
		L = L / dist; 
		vec3 L = normalize(L);
		NoL = dot(N,L);
		NdotL = clamp( NoL, 0.0, 1.0 );

		float att_factor = u_light_max_distance - dist;
		att_factor /= u_light_max_distance;
		att_factor = max(att_factor, 0.0);

		float min_angle_cos = u_light_cone_info.y;
		float max_angle_cos = u_light_cone_info.x;
		if (u_light_type == SPOTLIGHT){
			NdotL = 1.0;
			vec3 D = normalize(u_light_front);
			float cos_angle = dot( D, L );
			if( cos_angle < min_angle_cos  ){
	 			att_factor = 0.0;
			} else if ( cos_angle < max_angle_cos) {
				att_factor *= (cos_angle - min_angle_cos) / (max_angle_cos - min_angle_cos);
			}
			if(u_light_cast_shadow == 1)
				shadow_factor = computeShadow(v_world_position);
		}

		
		light_add = u_light_color * att_factor;
	} 
	
	vec3 H = normalize(V + L);
	float NoH = dot(N, H);
	float NoV = dot(N, V);
	float LoH = dot(L, H);

	//we compute the reflection in base to the color and the metalness
	vec3 f0 = mix( vec3(0.5), color.xyz, metalness );

	//metallic materials do not have diffuse
	vec3 diffuseColor = (1.0 - metalness) * color.xyz;

	//compute the specular
	vec3 Fr_d = specularBRDF(roughness, f0, NoH, NoV, NoL, LoH); 

	// Here we use the Burley, but you can replace it by the Lambert.
	float linearRoughness = roughness * roughness;
	//vec3 Fd_d = diffuseColor * Fd_Lambert(color.xyz); 
	vec3 Fd_d = diffuseColor * Fd_Burley(NoV,NoL,LoH,linearRoughness); 
	
	//add diffuse and specular reflection
	vec3 direct = Fr_d + Fd_d;
	if (u_PBR == 0){
		direct = vec3(1.0);
	}
	light_add *= shadow_factor * NdotL * direct;


\gbuffers.fs

#version 330 core

in vec3 v_position;
in vec3 v_world_position;
in vec3 v_normal;
in vec2 v_uv;

uniform vec4 u_color;
uniform sampler2D u_texture;
uniform sampler2D u_texture_emissive;
uniform sampler2D u_texture_normalmap;
uniform sampler2D u_texture_metallic_roughness;
uniform sampler2D u_texture_occlusion;
uniform float u_time;
uniform float u_alpha_cutoff;
uniform vec3 u_emissive_factor;
uniform bool u_norm_contr;

layout(location = 0) out vec4 FragColor;
layout(location = 1) out vec4 NormalColor;
layout(location = 2) out vec4 EmissiveOcclusion;
layout(location = 3) out vec4 ExtraColor;

#include "normalmap_functions"

void main()
{
	vec2 uv = v_uv;
	vec4 color = u_color;
	color *= texture( u_texture, uv );
	
	if(color.a < u_alpha_cutoff)
		discard;
	
	vec3 N = normalize(v_normal);
	vec3 normal_pixel = texture( u_texture_normalmap, uv ).xyz;
	if(!u_norm_contr){
    		N = perturbNormal(v_normal, v_world_position, v_uv, normal_pixel);
	}

	FragColor = vec4(color.xyz, texture( u_texture_metallic_roughness, uv ).z);
	NormalColor = vec4(N * 0.5 + vec3(0.5), texture( u_texture_metallic_roughness, uv ).y);

	EmissiveOcclusion = vec4(texture( u_texture_emissive, uv ).xyz * u_emissive_factor, texture( u_texture_occlusion, uv ).x);

	ExtraColor = vec4(fract(v_world_position), 1.0);
} 

\texture.fs

#version 330 core

in vec3 v_position;
in vec3 v_world_position;
in vec3 v_normal;
in vec2 v_uv;
in vec4 v_color;

uniform vec4 u_color;
uniform sampler2D u_texture;
uniform float u_time;
uniform float u_alpha_cutoff;

out vec4 FragColor;

void main()
{
	vec2 uv = v_uv;
	vec4 color = u_color;
	color *= texture( u_texture, v_uv );

	if(color.a < u_alpha_cutoff)
		discard;

	FragColor = color;
}

\ssao.fs

#version 330 core

in vec3 v_position;
in vec2 v_uv;

uniform sampler2D u_depth_texture;
uniform sampler2D u_normal_texture;
uniform mat4 u_inverse_viewprojection;
uniform mat4 u_viewprojection;
uniform vec2 u_iRes;
uniform float u_radius;
uniform vec3 u_points[64];
uniform float u_max_distance;

out vec4 FragColor;

void main()
{
	vec2 uv = gl_FragCoord.xy *u_iRes.xy;
	vec3 N = texture( u_normal_texture, v_uv ).xyz * 2.0 - vec3(1.0);
	N = normalize(N);
	float depth = texture( u_depth_texture, v_uv).x;
	if(depth == 1.0)
		discard;

	vec4 screen_pos = vec4(uv.x*2.0-1.0, uv.y*2.0-1.0, depth*2.0-1.0, 1.0);
	vec4 proj_worldpos = u_inverse_viewprojection * screen_pos;
	vec3 v_world_position = proj_worldpos.xyz / proj_worldpos.w;
	int num = 64;
	
	for(int i = 0; i < 64; ++i)
	{
		vec3 random_point = u_points[i];

		//check in which side of the normal
		if(dot(N,random_point) < 0.0)
			random_point *= -1.0;
		//vec3 p = v_world_position + u_points[i] * u_radius;
		vec3 p = v_world_position + random_point * u_radius;
		vec4 proj = u_viewprojection * vec4(p, 1.0);
		proj.xy /= proj.w; //convert to clipspace from homogeneous
		//apply a tiny bias to its z before converting to clip-space
		proj.z = (proj.z - 0.005) / proj.w;
		proj.xyz = proj.xyz * 0.5 + vec3(0.5); //to [0..1]

		//read p true depth
		float pdepth = texture( u_depth_texture, proj.xy ).x;
		//compare true depth with its depth
		if( pdepth < proj.z) //if true depth smaller, is inside
			num--; //remove this point from the list of visible
	}

	float ao = float(num) / 64.0;
	FragColor = vec4(ao, ao, ao, 1.0); 
}

\deferred_global.fs

#version 330 core

in vec3 v_position;
in vec2 v_uv;
in vec3 v_world_position;
in vec3 v_normal;

uniform sampler2D u_color_texture;
uniform sampler2D u_normal_texture;
uniform sampler2D u_depth_texture;
uniform sampler2D u_emissive_occlusion_texture;
//uniform sampler2D u_normalmap_texture;
uniform sampler2D u_normalmap_texture;
uniform sampler2D u_ao_texture;

uniform vec3 u_ambient_light;

uniform vec3 u_light_position;
uniform vec3 u_light_color;
uniform int u_light_type;
uniform vec3 u_light_front;
uniform vec2 u_light_cone_info;
uniform float u_light_max_distance;
uniform vec3 u_emissive_first;

uniform mat4 u_inverse_viewprojection;
uniform vec2 u_iRes;
uniform vec3 u_camera_pos;

uniform int u_PBR;

#define POINTLIGHT 1
#define SPOTLIGHT 2
#define DIRECTIONALLIGHT 3

out vec4 FragColor;
out float glFragDepth;

#include "ComputeShadow"

#include "specullar_function"

#include "normalmap_functions"

void main()
{
	vec2 uv = gl_FragCoord.xy *u_iRes.xy;
	float depth = texture( u_depth_texture, uv).x;

	if(depth == 1.0)
		discard;

	vec4 GB0 = texture( u_color_texture, uv );
	vec4 GB1 = texture( u_normal_texture, uv );
	vec3 color = GB0.xyz;
	float metalness = GB0.a;
	float roughness = GB1.a;

	vec3 emissive = texture(u_emissive_occlusion_texture, uv).xyz;
	float occlusion = texture(u_emissive_occlusion_texture, uv).w; 

	float ao_factor = texture( u_ao_texture, uv ).x;

	//ao_factor = pow( ao_factor, 3.0 );

	vec3 light = u_ambient_light * occlusion * ao_factor;

	vec4 screen_pos = vec4(uv.x*2.0-1.0, uv.y*2.0-1.0, depth*2.0-1.0, 1.0);
	vec4 proj_worldpos = u_inverse_viewprojection * screen_pos;
	vec3 v_world_position = proj_worldpos.xyz / proj_worldpos.w;

	vec3 L;
	vec3 normal = GB1.xyz * 2.0 - vec3(1.0);
	vec3 N = normalize(normal);
	
	vec3 V = normalize(u_camera_pos - v_world_position);

	float NdotL = 0.0;
	
	#include "ComputeLights"

	vec3 final_color;

	final_color = ((NdotL * light_add) + light) * color.xyz + emissive * u_emissive_first;

	FragColor = vec4(final_color, 1.0);
	glFragDepth = depth;

}


\light.fs

#version 330 core

in vec3 v_position;
in vec3 v_world_position;
in vec3 v_normal;
in vec2 v_uv;
in vec4 v_color;

uniform vec4 u_color;
uniform sampler2D u_texture_albedo;
uniform sampler2D u_texture_emissive;
uniform sampler2D u_texture_occlusion;
uniform sampler2D u_texture_normalmap;
uniform sampler2D u_texture_metallic_roughness; //Still not used
uniform float u_time;
uniform float u_alpha_cutoff;
uniform bool u_norm_contr;
uniform vec3 u_camera_pos;

uniform vec3 u_ambient_light;
uniform vec3 u_emissive_factor;
uniform vec3 u_light_position;
uniform vec3 u_light_color;
uniform vec3 u_light_front;
uniform float u_light_max_distance;
uniform vec2 u_light_cone_info;

uniform int u_light_type;
uniform int u_PBR;

#define POINTLIGHT 1
#define SPOTLIGHT 2
#define DIRECTIONALLIGHT 3

out vec4 FragColor;

#include "ComputeShadow"

#include "specullar_function"

#include "normalmap_functions"

void main()
{
	vec2 uv = v_uv;
	vec4 color = u_color;
	color *= texture( u_texture_albedo, v_uv );
	
	float metalness = texture(u_texture_metallic_roughness, v_uv).z;
	float roughness = texture(u_texture_metallic_roughness, v_uv).y;

	if(color.a < u_alpha_cutoff)
		discard;

	vec3 light = u_ambient_light * texture(u_texture_occlusion, v_uv).x;
	
	vec3 L;
	vec3 N = normalize(v_normal);
	vec3 normal = texture(u_texture_normalmap, v_uv).xyz;
	if(!u_norm_contr){
    		N = perturbNormal(v_normal, v_world_position, v_uv, normal);
	}
	float NdotL = 0.0;
	vec3 V = normalize(u_camera_pos - v_world_position);

	#include "ComputeLights"

	light += (NdotL * light_add);

	vec4 final_color;
	final_color.xyz = (color.xyz * light) + u_emissive_factor * texture( u_texture_emissive, v_uv ).xyz;
	final_color.a = color.a;
	
	FragColor = final_color;
}


\skybox.fs

#version 330 core

in vec3 v_position;
in vec3 v_world_position;

uniform samplerCube u_texture;
uniform vec3 u_camera_position;
out vec4 FragColor;

void main()
{
	vec3 E = v_world_position - u_camera_position;
	vec4 color = texture( u_texture, E );
	FragColor = color;
}


\multi.fs

#version 330 core

in vec3 v_position;
in vec3 v_world_position;
in vec3 v_normal;
in vec2 v_uv;

uniform vec4 u_color;
uniform sampler2D u_texture;
uniform float u_time;
uniform float u_alpha_cutoff;

layout(location = 0) out vec4 FragColor;
layout(location = 1) out vec4 NormalColor;

void main()
{
	vec2 uv = v_uv;
	vec4 color = u_color;
	color *= texture( u_texture, uv );

	if(color.a < u_alpha_cutoff)
		discard;

	vec3 N = normalize(v_normal);

	FragColor = color;
	NormalColor = vec4(N,1.0);
}


\depth.fs

#version 330 core

uniform vec2 u_camera_nearfar;
uniform sampler2D u_texture; //depth map
in vec2 v_uv;
out vec4 FragColor;

void main()
{
	float n = u_camera_nearfar.x;
	float f = u_camera_nearfar.y;
	float z = texture2D(u_texture,v_uv).x;
	if( n == 0.0 && f == 1.0 )
		FragColor = vec4(z);
	else
		FragColor = vec4( n * (z + 1.0) / (f + n - z * (f - n)) );
}


\instanced.vs

#version 330 core

in vec3 a_vertex;
in vec3 a_normal;
in vec2 a_coord;

in mat4 u_model;

uniform vec3 u_camera_pos;

uniform mat4 u_viewprojection;

//this will store the color for the pixel shader
out vec3 v_position;
out vec3 v_world_position;
out vec3 v_normal;
out vec2 v_uv;

void main()
{	
	//calcule the normal in camera space (the NormalMatrix is like ViewMatrix but without traslation)
	v_normal = (u_model * vec4( a_normal, 0.0) ).xyz;
	
	//calcule the vertex in object space
	v_position = a_vertex;
	v_world_position = (u_model * vec4( a_vertex, 1.0) ).xyz;
	
	//store the texture coordinates
	v_uv = a_coord;

	//calcule the position of the vertex using the matrices
	gl_Position = u_viewprojection * vec4( v_world_position, 1.0 );
}