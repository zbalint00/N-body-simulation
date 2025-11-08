#version 150

in vec4 vs_in_pos;
 
out Vertex
{
	vec4 color;
} vertex;
 
void main()
{
	vec2 pos = vs_in_pos.xy;
    vec2 vel = vs_in_pos.zw;
	gl_Position = vec4(pos, 0, 1);
    
	 // compute speed magnitude
    float speed = length(vel);
    float maxSpeed = 4.0; 
    float t = clamp(speed / maxSpeed, 0.0, 1.0);
    vec3 color = mix(vec3(1.0,1.0,1.0), vec3(1.0,0.0,0.0), t);
    vertex.color = vec4(color, 1.0);
}