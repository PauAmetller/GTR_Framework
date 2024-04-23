//example of some shaders compiled
flat basic.vs flat.fs
texture basic.vs texture.fs
light basic.vs light.fs
skybox basic.vs skybox.fs
depth quad.vs depth.fs
multi basic.vs multi.fs

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

//Shadow_map resources
uniform int u_light_cast_shadow;
uniform sampler2D u_shadow_map;
uniform mat4 u_shadow_map_view_projection;
uniform float u_shadow_bias;

uniform vec3 u_ambient_light;
uniform vec3 u_emissive_factor;
uniform vec3 u_light_position;
uniform vec3 u_light_color;
uniform vec3 u_light_front;
uniform float u_light_max_distance;
uniform vec2 u_light_cone_info;

uniform int u_light_type;

#define POINTLIGHT 1
#define SPOTLIGHT 2
#define DIRECTIONALLIGHT 3

out vec4 FragColor;

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

	//read depth from depth buffer in [0..+1] non-linear
	float shadow_depth = texture( u_shadow_map, shadow_uv).x;

	//compute final shadow factor by comparing
	float shadow_factor = 1.0;

	//we can compare them, even if they are not linear
	if( shadow_depth < real_depth )
		shadow_factor = 0.0;
	return shadow_factor;

}

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


void main()
{
	vec2 uv = v_uv;
	vec4 color = u_color;
	color *= texture( u_texture_albedo, v_uv );

	if(color.a < u_alpha_cutoff)
		discard;

	vec3 light = u_ambient_light * texture(u_texture_occlusion, v_uv).xyz;
	
	vec3 L;

	vec3 light_add;
	float shadow_factor = 1.0;
	if(u_light_cast_shadow == 1)
		shadow_factor = computeShadow(v_world_position);
	if ( u_light_type == DIRECTIONALLIGHT)
	{
		L = u_light_front;
		light_add = u_light_color;
	}
	else if (u_light_type == SPOTLIGHT || u_light_type == POINTLIGHT) //spot and point
	{
		L = u_light_position - v_world_position;
		float dist = length(L);

		float min_angle_cos = u_light_cone_info.x;
		float max_angle_cos = u_light_cone_info.y;
		float spot_factor = 1.0;
		if (u_light_type == SPOTLIGHT){
			vec3 L_norm = normalize(L);
			vec3 D = normalize(u_light_front);
			float cos_angle = dot( D, L_norm );
			if( cos_angle < min_angle_cos  ){
	 			spot_factor = 0.0;
			} else if ( cos_angle < max_angle_cos) {
				spot_factor *= (cos_angle - min_angle_cos) / (max_angle_cos - min_angle_cos);
			}
		}

		float att_factor = u_light_max_distance - dist;
		att_factor /= u_light_max_distance;
		att_factor = max(att_factor, 0.0);
		
		light_add = u_light_color * att_factor * spot_factor;
	} 

	vec3 normal = texture(u_texture_normalmap, v_uv).xyz;
    	vec3 perturbed_normal = perturbNormal(v_normal, v_world_position, v_uv, normal);
	float NdotL = clamp(max(dot(perturbed_normal, L), 0.0), 0.0, 1.0);

	light += (NdotL * light_add  * shadow_factor);

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