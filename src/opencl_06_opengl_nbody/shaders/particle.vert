#version 150

in vec3 vs_in_pos;
in vec3 vs_in_vel;
uniform mat4 u_viewProj;

out Vertex
{
	vec4 color;
} vertex;
 
void main()
{
	vec3 pos = vs_in_pos.xyz;
    vec3 vel = vs_in_vel.xyz;
	gl_Position = u_viewProj * vec4(pos, 1.0);
    
	// compute speed magnitude
    float speed = length(vel);
    float maxSpeed = 4.0; 
    float t = clamp(speed / maxSpeed, 0.0, 1.0);
    vec3 color = mix(vec3(1.0,1.0,1.0), vec3(1.0,0.0,0.0), t);
    vertex.color = vec4(color, 1.0);
}