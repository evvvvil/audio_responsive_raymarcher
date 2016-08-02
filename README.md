# audio_responsive_raymarcher
evvvvil's raymarching distance fields audio-responsive mega shader

This shader is a raymarcher rendering distance fields equations.
It basically generates live 3d geometry on the GPU pipeline in a totally procedural manner and without using any vertices or mesh.
The resulting geometry is displaced, lit, shaded and has depth and therefore can be mixed in a scene with normal polygonal geometry.

Two different shapes can be created with this shader: one hellish fireball with very high details and audio-responsive perlin displacement mapping (please see screenshots in folder called "raymarch_hell01.png", etc...)
The other shape is an intricate procedural column aka the "flash-bang tunnel, broh" and can be viewed in its full-on glory as a gif here:
http://imgur.com/31GnVp4 and http://imgur.com/K2rgATV

