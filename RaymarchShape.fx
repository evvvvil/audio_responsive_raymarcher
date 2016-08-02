#define TWOPI 6.28318531;
#define PI 3.14159265;
cbuffer cbControls:register(b0){
	float4x4 tWVP:WORLDVIEWPROJECTION;
	float4x4 tVP:VIEWPROJECTION;
	float4x4 tW:WORLD;
	float4x4 tV:VIEW;
	float4x4 tP:PROJECTION;
	float4x4 tWI:WORLDINVERSE;
	float4x4 tVI:VIEWINVERSE;
	float4x4 tPI:PROJECTIONINVERSE;
	float4x4 tWIT:WORLDINVERSETRANSPOSE;
	float2 R:TARGETSIZE;
	static const float limito = 15.0;
	static const float3 zeroZeroZero = {0.0,0.0,0.0};
	static const float3 constBoxSize = {30.0,4.0,30.0};
	static const float2 cylinderSize = {0.3,1.6};
	static const float3 sizeCube ={0.8,0.5,0.5};
	static const float3 cubeOffset={0.35,0.2,0.2};
	static const float3 repeaterConst={15.0,70.0,15.0};
	
	float dMulti=1.3;
	
	//HELL SHAPE VARIABLES:
	float time<String uiname="Sphere Time";> = 6.4;
	float scale<String uiname="Sphere Scale";> = 5.0;
	float dispMultiplerX=1.14;
	float dispMultiplerY=8.0;
	float dispMultiplerZ=3.24;
	float dispMultiplerXSinY=5.18;
	float dispMultiplerYSinZ=1.62;
	float dispMultiplerZSinX=6.46;
	float dispAmount=0.06;
	float xOffset=1.78;
	static float2 uv;	
	float dispMappingMultiplier=0.0;
	float3 positionOffset ={0.0,0.0,0.0};
	float positionMultiplier = 0.5;
	//COLUMN VARIABLES
	float noiseAmount=0;
	float3 noiseOffset1={0.0,0.0,0.0};
	float3 noiseOffset2={5.2,16.5,100.2};
	float3 noiseOffset3={15.3,0.5,1.2};
	
	//DIFFUSE VARIABLES:
	//float4 matcolor <bool color=true; string uiname="Material Colour";>  = {0.2, 0.1, 1, 1};
	//float4 lightcolor <bool color=true; string uiname="Light Colour";> = {0.2,0.1,1,1};
	
	//TEXTURE PATTERN VARIABLES:
	float freq<String uiname="Texture Frequency";> = 5;
	float thin<String uiname="Texture Thickness";> = 4;
	float mult<String uiname="Texture Cos Mult";> =0;
	float mult2<String uiname="Texture Cos Mult2";> =0;
	float offsetMult<String uiname="Texture Cos Offset";> =0;
	float offsetMult2<String uiname="Texture Cos Offset2";> =0;
	float timo<String uiname="Texture Time";> =0;
};
//PHONG DIRECTIONAL VARIABLES
cbuffer cbLightData : register(b3)
{
	//light properties
	float3 lDir <string uiname="Light Direction";> = {0, -5, 2};        //light direction in world space
	float4 lAmb  <bool color=true; String uiname="Ambient Color";>  = {0.15, 0.15, 0.15, 1};
	float4 lDiff <bool color=true; String uiname="Diffuse Color";>  = {0.85, 0.85, 0.85, 1};
	float4 lSpec <bool color=true; String uiname="Specular Color";> = {0.35, 0.35, 0.35, 1};
	float lPower <String uiname="Power"; float uimin=0.0;> = 25.0;     //shininess of specular highlight
}

bool hellShape=true;

//HELL SHAPE TEXTURES AND SAMPLERS:
Texture2D texDisp <string uiname="Displacement Texture";>;
Texture2D texColor <string uiname="Color Texture";>;
SamplerState mySampler : IMMUTABLE
{
	Filter = MIN_MAG_MIP_LINEAR;
	AddressU =Clamp;
	AddressV = Clamp;
};
SamplerState wrapSampler : IMMUTABLE
{
	Filter = MIN_MAG_MIP_LINEAR;
	AddressU = Wrap;
	AddressV = Wrap;
};

struct VS_IN{float4 PosO:POSITION;float4 TexCd:TEXCOORD0;};
struct VS_OUT{float4 PosWVP:SV_POSITION;float4 TexCd:TEXCOORD0;};
VS_OUT VS(VS_IN In){VS_OUT Out=(VS_OUT)0;Out.TexCd=In.TexCd;Out.PosWVP=mul(float4(In.PosO.xy,0,1),tW);return Out;}

//phong directional function
float4 PhongDirectional(float3 NormV, float3 ViewDirV, float3 LightDirV)
{
	float4 amb = float4(lAmb.rgb, 1);
    //halfvector
    float3 H = normalize(ViewDirV + LightDirV);
    //compute blinn lighting
    float3 shades = lit(dot(NormV, LightDirV), dot(NormV, H), lPower).rgb;
    float4 diff = lDiff * shades.y;
    diff.a = 1;
    //reflection vector (view space)
    float3 R = normalize(2 * dot(NormV, LightDirV) * NormV - LightDirV);
    //normalized view direction (view space)
    float3 V = normalize(ViewDirV);
    //calculate specular light
    float4 spec = pow(max(dot(R, V),0), lPower*.2) * lSpec;
    return (amb + diff) + spec;
}

float3 UVtoEYE(float2 UV){return normalize(mul(float4(mul(float4((UV.xy*2-1)*float2(1,-1),0,1),tPI).xy,1,0),tVI).xyz);}
float2 r2d(float2 x,float a){a*=acos(-1)*2;return float2(cos(a)*x.x+sin(a)*x.y,cos(a)*x.y-sin(a)*x.x);}
float3 r3d(float3 p,float3 z){z*=acos(-1)*2;float3 x=cos(z),y=sin(z);return mul(p,float3x3(x.y*x.z+y.x*y.y*y.z,-x.x*y.z,y.x*x.y*y.z-y.y*x.z,x.y*y.z-y.x*y.y*x.z,x.x*x.z,-y.y*y.z-y.x*x.y*x.z,x.x*y.y,y.x,x.x*x.y));}
float3x3 lookat(float3 dir,float3 up=float3(0,1,0)){float3 z=normalize(dir);float3 x=normalize(cross(up,z));float3 y=normalize(cross(z,x));return float3x3(x,y,z);} 

float smin(float a, float b,float k){
	float res = exp (-k*a)+exp(-k*b);
	return -log(res)/k;
}

float sdSphere( float3 p, float s )
{
  return length(p)-s;
}

float hash( float n ) { return frac(sin(n)*43758.5453123); }

float sdCappedCylinder( float3 p, float2 h )
{
  float2 d = abs(float2(length(p.xz),p.y)) - h;
  return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

float noise(float3 x )
{
    float3 p = floor(x);
    float3 f = frac(x);
    f = f*f*(3.0-2.0*f);
	
    float n = p.x + p.y*157.0 + 113.0*p.z;
    return lerp(lerp(lerp( hash(n+  0.0), hash(n+  1.0),f.x),
                   lerp( hash(n+157.0), hash(n+158.0),f.x),f.y),
               lerp(lerp( hash(n+113.0), hash(n+114.0),f.x),
                   lerp( hash(n+270.0), hash(n+271.0),f.x),f.y),f.z);
}

float simpleDisplace(float3 p, float primitive, float displacement){
	return primitive+displacement;
}

float sdBox( float3 p, float3 b )
{
  float3 d = abs(p) - b;
  return min(max(d.x,max(d.y,d.z)),0.0) +
         length(max(d,0.0));
}

float boxesBro( float3 p, float3 b,float3 offset )
{
	//float d=boxRepeater(p,float3(2,2,12),b);
	float d=sdBox(p,b);
	d=min(d,sdBox(p+offset,b));
	//d=min(d,sdBox(p+offset*2,b));
	//d=min(d,sdBox(p-offset,b));
	d=min(d,sdCappedCylinder(p,cylinderSize));
	return d;
}

float shapeYo(float3 newP,float3 newP2,float3 newP3,float3 sizeCube,float3 cubeOffset){
	return min(boxesBro(newP,sizeCube,cubeOffset),min(boxesBro(newP2,sizeCube,cubeOffset),boxesBro(newP3,sizeCube,cubeOffset)));
}
float shapeRepeater(float3 p, float3 c, float3 sizeCube,float3 cubeOffset){
	//float3 q=fmod(p,c)-0.5*c;
	float3 q=p;
	float3 noiseP = noise(q+noiseOffset1)*noiseAmount;
	float3 noiseP2 = noise(q+noiseOffset2)*noiseAmount;
	float3 noiseP3 = noise(q+noiseOffset3)*noiseAmount;
	float finalSin = 1;//sin(q.y*0.10f);
	//if(finalSin<0.3){
		//finalSin=0.3;
	//}
	float3 newP=float3(q.x+noiseP.x,noiseP.y,q.z+noiseP.z)*finalSin;
	float3 newP2=float3(q.x+noiseP2.x,noiseP2.y,q.z+noiseP2.z);
	float3 newP3=float3(q.x+noiseP3.x,noiseP3.y,q.z+noiseP3.z);
	return shapeYo(newP,newP2,newP3,sizeCube,cubeOffset);
	
}
/* MATERIAL ID Thingy...
float2 f2(float3 p){
	float2 d={9999999,1};	
	//This is bit overkill maybe:
	if(p.x>limito||p.x<-limito||p.y>limito||p.y<-limito||p.z>limito||p.z<-limito){
		return 1;
	}
	d=float2(shapeRepeater(p,repeaterConst,sizeCube,cubeOffset),1);
	//SHAPE TWO - HELL:
	//p+=float3(0.0,10.0,0.0);
	p*=0.5;
	float disp =sin(cos(time)*dispMultiplerX*p.x+sin(p.y)*dispMultiplerXSinY)+sin(cos(time)*dispMultiplerY*p.y+sin(p.z)*dispMultiplerYSinZ)+sin(cos(time)*dispMultiplerZ*p.z+xOffset+sin(p.x)*dispMultiplerZSinX);

	//uv mapping
    float3 pnorm = normalize(p);
    uv = float2(0.0,0.0);
    uv.x = 0.5 + atan2(pnorm.z, pnorm.x) / (2.*3.14159f);
    uv.y = 0.5 - asin(pnorm.y) / 3.14159f;
    
    float y = texDisp.SampleLevel(mySampler,uv,0).r;
    float y2 = dispMappingMultiplier * y;
	
	float2 d2 = float2(simpleDisplace(p,sdSphere(p/scale , 0.5 + y2)*scale,disp*dispAmount),2);
	
	if (d2.x < d.x) d = d2;
	return d;
}*/
float f(float3 p){
	float d=9999999.0;
	if(hellShape){		
		//SHAPE ONE - HELL:
		p+=positionOffset;
		p*=positionMultiplier;
		float disp =sin(cos(time)*dispMultiplerX*p.x+sin(p.y)*dispMultiplerXSinY)+sin(cos(time)*dispMultiplerY*p.y+sin(p.z)*dispMultiplerYSinZ)+sin(cos(time)*dispMultiplerZ*p.z+xOffset+sin(p.x)*dispMultiplerZSinX);		
		//uv mapping
		float3 pnorm = normalize(p);
		uv = float2(0.0,0.0);
		uv.x = 0.5 + atan2(pnorm.z, pnorm.x) / TWOPI;
		uv.y = 0.5 - asin(pnorm.y) / PI;
		
		float y = texDisp.SampleLevel(mySampler,uv,0).r;
		float y2 = dispMappingMultiplier*y;
		
		d = simpleDisplace(p,sdSphere(p/scale , 0.5 + y2)*scale,disp*dispAmount);
		d*=dMulti;
		return d;
	}else{
		//SHAPE TWO - COLUMN:
		if(p.x>limito||p.x<-limito||p.y>limito||p.y<-limito||p.z>limito||p.z<-limito){
			return 1;
		}		
		d=shapeRepeater(p,repeaterConst,sizeCube,cubeOffset);
		d*=dMulti;
		return d;
	}

	
}
float AmbientOcclusion(float3 p,float3 n,float scale=1){
	float ao=1;
	float g=f(p);
	float shd=0;
	int iter=3;
	for(int i=0;i<iter;i++){
		float ff=scale;
		ff*=pow(2,12*pow((float)i/iter,2));
		float smp=max(0,1-f(p+n*ff)/ff);
		shd+=pow(smp,2)/iter*pow(0.5,(float)i/iter);
	}
	ao=1-shd;
	ao=saturate(ao);
	return ao;
}

//COLOR SATURATION VARIABLE AND FUNCTION

//static const float4 coeff = {0.3086, 0.6094, 0.0820, 0};
/*float3 sat <string uiname="Saturation";> = 1.;
float4 Saturation(float4 Color) {
	float4 _s = (1.-sat.rgbb)*coeff;
	
	float4x4 mat = float4x4(_s.r+sat.r,_s.r,_s.r,0,
							_s.g,_s.g+sat.g,_s.g,0,
							_s.b,_s.b,_s.b+sat.b,0,
							0,0,0,1);
	return mul(Color,mat);
}*/


struct PS_OUT{
	float4 Color:SV_TARGET;
	float Depth:SV_DEPTH;
};
PS_OUT PS(VS_OUT In){
	float2 UV=In.TexCd.xy;
	float3 cd=UVtoEYE(UV);
	float3 cp=mul(float4(0,0,0,1),tVI).xyz;
	
	float3 p=cp;
	float z=length(p-cp);
	float3 Norm=0;
	float d;

	for(int i=0;i<120;i++){
		//UGLY BUT FAST WAY OF STOPPING RAY
		/*if(!hellShape && (p.x>limito||p.x<-limito||p.y>limito||p.y<-limito||p.z>limito||p.z<-limito)){
			break;
		}*/

		d=f(p);
		
		p+=cd*d.x;
		
		z=length(p-cp);
		
		if(abs(d.x)<.002*z)break;
	}
	z=length(p-cp);
	float ff=f(p);
	if(abs(ff)>0.10)discard;
	/*if(!hellShape && (p.x>limito||p.x<-limito||p.y>limito||p.y<-limito||p.z>limito||p.z<-limito)){
			discard;
		}*/
	//THIS SUPER SLOW WAY OF STOPING RAY:
	//if(abs(ff)>.5||distance(p,ff)>limito)discard;
	
	//NORMALS:
	float2 e={.0001*sqrt(z),0};
	Norm=normalize(float3(f(p+e.xyy),f(p+e.yxy),f(p+e.yyx))-f(p));
	
	//MATERIAL ID SYSTEM pass again p this time in float2 simplified f2,
	//rather than float f, just to get material ID the rest we don't give a shit...
	//This is a bit overkill maybe but thinking performance wise
	//it might be faster to have clone of f function called f2 just for that
	//purpose, as f is called often (float2 vs float)
	//float materialID=f2(p).y;
	
	//LIGHTING:	
	float4 c=1;
	
	//SIMPLE FRONTAL LIGHTING
	//c.rgb=Norm*0.5+0.5;
	
	//DIFFUSE LIGHTING:
	//c= saturate(-dot(lDir, Norm)) * lightcolor * matcolor;
	
	//SATURATE HERE BRO
	//c=Saturation(c);	
	
	//GREYSCALE
	//float g=saturate(-dot(cd,Norm));
	//c.rgb=g;
	
	//MATERIAL ONE (COLUMN):
	
	//if(materialID<2){
	if(!hellShape){
		float3 LightDirV = normalize(-mul(float4(lDir,0.0f), mul(tW,tV)).xyz);
		float3 NormV = normalize(mul(mul(Norm, (float3x3)tWIT),(float3x3)tV).xyz);
		c.rgb = PhongDirectional(NormV, cd, LightDirV).xyz;	
	    //GOOD PATTERN!
		c.r*=-(ceil(sin(p.x-1*cos(freq*p.x*5)+timo+ceil(cos(p.y*mult+offsetMult)))-thin)+ceil(sin(p.y*mult2+offsetMult2)));
	}else{	
		//MATERIAL TWO (HELL):		
		//c.rgb=texColor.Sample(wrapSampler,uv*2).rgb;
		//c.rgb=texColor.Load(uint3(uv*2,1)).rgb;
		c.rgb=texColor.SampleLevel(wrapSampler,uv*2,0).rgb;
	}	
	//SUBSTRACTIVE AMBIENT OCCLUSION BROH!!!
	c.rgb-=AmbientOcclusion(p,Norm,.05);
	
	float4 PosWVP=mul(float4(p.xyz,1),tVP);
	PS_OUT Out;
	Out.Color=c;
	Out.Depth=PosWVP.z/PosWVP.w;
	return Out;
}
float4 PSc(VS_OUT In): SV_Target{
	PS_OUT Out=PS(In);
	return Out.Color;
}
technique10 Color{
	pass P0{
		SetVertexShader(CompileShader(vs_5_0,VS()));
		SetPixelShader(CompileShader(ps_5_0,PSc()));
	}
}
technique10 ColorAndDepth{
	pass P0{
		SetVertexShader(CompileShader(vs_5_0,VS()));
		SetPixelShader(CompileShader(ps_5_0,PS()));
	}
}




