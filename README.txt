ORIOL SOLER GONZALEZ 255525 oriol.soler02@estudiant.upf.edu
PAU AMETLLER LÓPEZ 254537 pau.ametller01@estudiant.upf.edu


--- GTR ASSIGNMENT 2 ---
En aquesta segona entrega, hem implementat el mode de renderitzat de Deferred.

Primer de tot, per intentar assimilar tant com fos possible el resultat de Deferred
al que ja teníem (Forward), hem renderitzat l'escena amb totes les textures prèvies
(color, normals, emissive, occlusion, metal roughness...). Aquest cas però, hem
utilitzat GBuffers per guardar les textures en un sol objecte. Al voler tenir totes
les textures en un sol GBuffer, hem necessitat crear un 4 component d'aquest i al final
hem distribuït el GBuffer de forma que:
0 - Color.xyz + Metal Roughness.z     (Color + metalness)
1 - NormalColor + Metal Roughness.y   (Normal(Normalmap applied) + roughness)
2 - Emissive.xyz + Occlusion.x        (Emissive + Occlusion)
3 - Extra Color                       (Fractioned world position)
Es poden consultar aquestes textures en el menú de GBuffers de l'editor i jugar amb treure i posar 
textures i llums des dels desplegables de Texture Options i Light Options.

Un cop renderitzada l'escena amb forward, renderitzem els elements alpha amb forward,
mètode que ja teníem implementat. Observem, però que costa molt veure aquests objectes
i que només els podem percebre observant des de molt a prop i a vegades, des de dins el 
mateix cotxe. Per les llums, estàvem creant quads a pantalla completa per a cada una, 
així que pels spots i pointlights hem construït esferes on només a dintre es realitzaran
els càlculs per a aquestes llums.

Pel PBR hem calculat el V (vector towards the eye) i H (half vector between V and L), juntament amb la
L (vector toward the light) i la N (normal vector at the point), les qual ja haviem obtingut previament 
per calcular la llum, hem fet els seus dot products i applicat les formules proporcionades a les slides.
Poden observar que tot i que enfosqueix la scena, ja que fa que només reflexi una part de la llum depenent
de les propietats del material, fa que algun objectes metallics com els retrovisors es noti un llum molt 
més realista.
Simplement té una checkbox per activar-lo o desactivar-lo, dins de Texture OPTIONS.

Després, hem aplicat l'algoritme del SSAO per poder afegir una ambient oclusion més
fidel a l'escena. Per fer això, ens centrem a generar punts aleatoris dintre d'una esfera,
i observar la depth d'aquests. A partir d'això, com més pròxims eren els altres elements
de l'escena, més oclusió dibuixem. Tot i així, aquest mètode només es centrava en el
depth buffer sense tenir en compte l'escena amb les seves normals. Així, l'ampliem per 
generar l'algoritme conegut com a SSAO+, el qual agafa com a referència la normal dels
objectes i orienta els punts només en l'hemisferi que marca aquesta normal. El resultat amb això 
és molt més precís i podem observar oclusions més encertades. Tot i així, encara observàvem
artefactes com línies entre diferents zones d'oclusió i és per això que hem implementat una rotació.

Tant les textures de SSAO i SSAO+ soles com en combinació amb la resta d'escena es pot visualitzar
en el SSAO OPTIONS, on hi ha un desplegable per escollir entre el bàsic i el +, una opció per visualitzar
la textura sola i una altre per desactivar-la en l'escena. Finalment, podem regular el resultat controlant
el radi de l'esfera dels random points i la màxima distància de oclusió (tot i que necessita valors baixos,
molt útils per evitar falses "aures" d'oclusió).

També tenim l'opció de emborronar la textura del SSAO amb un kernel de AxA, on la A es pot seleccionar des de
la opció de Kernel Size (amb un màxim de 5). Aquesta opció l'hem desenvolupada a partir d'aplicar un filtre gaussià
que a partir d'una textura i un kernel de AxA, té en compte els píxels del kernel per amitjanar el pixel central
a partir d'uns pesos amb distribució gaussiana.

Hem pasat les textures que contenien colors com la color i emissive, the gamma a linear la començament del
fragment shader i hem retornat el color resultant a gamma abans de retornar el FragColor.

Després, hem applicat el tonemapper el qual té un menú desplegable en la pestanya del rendering desde on
es pot activar o desactivar, i modificar els parametres per aconseguir una escena més semblan a com 
nosaltres persebriem la realitat. 

