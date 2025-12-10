#version 150
 
layout (points) in;
layout (triangle_strip) out;
layout (max_vertices = 4) out;    
 
uniform float particle_size = 0.01;
 
in Vertex
{
	vec4 color;
} vertex[];
 
out vec2 Vertex_UV;
out vec4 Vertex_Color;
   
void main(void)
{
	vec4 P = gl_in[0].gl_Position;
    vec4 ndc = P / P.w; // Normalized Device Coordinates
    float hs = 0.5 * particle_size;

    // Order: a (left-bottom), b (left-top), d (right-bottom), c (right-top)
	vec2 offsets[4] = vec2[](
        vec2(-hs, -hs), // a
        vec2(-hs,  hs), // b
        vec2( hs, -hs), // d
        vec2( hs,  hs)  // c
    );

    vec2 uvs[4] = vec2[](
        vec2(0.0, 0.0), // a
        vec2(0.0, 1.0), // b
        vec2(1.0, 0.0), // d
        vec2(1.0, 1.0)  // c
    );

    for (int i = 0; i < 4; ++i)
    {
        vec4 outNDC = vec4(ndc.xy + offsets[i], ndc.z, 1.0);
        gl_Position = outNDC * P.w;
        Vertex_UV    = uvs[i];
        Vertex_Color = vertex[0].color;

        EmitVertex();
    }

    EndPrimitive();
	
}