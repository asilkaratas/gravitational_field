#define STRINGIFY(A) #A

// vertex shader
const char *vertexShader = STRINGIFY(
                               uniform float pointRadius;  // point size in world space
                               uniform float pointScale;   // scale to calculate size in pixels
                               uniform float densityScale;
                               uniform float densityOffset;
                               void main()
{

    // calculate window-space point size

    vec3 posEye = vec3(gl_ModelViewMatrix * vec4(gl_Vertex.xyz, 1.0));
    float dist = length(posEye);
    gl_PointSize = pointRadius * (pointScale / dist);

    gl_TexCoord[0] = gl_MultiTexCoord0;
    gl_Position = gl_ModelViewProjectionMatrix * vec4(gl_Vertex.xyz, 1.0);

    gl_FrontColor = gl_Color;



}
                           );


const char *spherePixelShader = STRINGIFY(
                                    void main()
{
	vec3 N;
    N.xy = gl_TexCoord[0].xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    float mag = dot(N.xy, N.xy);

    if (mag > 1.0) discard;

    gl_FragColor = gl_Color;
}
                                );
