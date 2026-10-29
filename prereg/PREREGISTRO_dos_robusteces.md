# Pre-registro — dos chequeos de robustez sobre resultados ya obtenidos

**Fecha:** 2026-10-29 · **Base:** `axisdynamics/lsgot_experiments_v4`, commit `6664c0c`

---

## Qué protege este documento, y qué no

**Los dos experimentos ya corrieron y sus resultados están reportados.** Este pre-registro **no**
puede proteger contra elegir el análisis después de ver los datos: para E26 y E20 eso ya pasó, y
sus veredictos están escritos y fechados.

Lo que sí es genuino, y es todo lo que este documento reclama: **las dos cantidades de abajo no
han sido calculadas.** La regla de qué significa cada salida se fija acá, antes de calcularlas.
Sin esto, las dos tienen la misma forma de trampa: un resultado conocido, una variante nueva, y
la libertad de quedarse con la que convenga.

---

## A · E26 · la tercera codificación de «abre con ritual»

### Estado

El veredicto de E26 —**el ancla no viene del plan de apertura**— está obtenido bajo **dos**
codificaciones independientes:

| Codificación | ritual / no | d en v̂ | p |
|---|---|---|---|
| Jorge, a mano, reconstruida por mí de su tabla | 11 / 9 | +0.545 | 0.248 |
| Claude, automática, radical en los primeros 120 caracteres | 13 / 7 | +0.508 | 0.300 |

### Qué falta

Jorge dio después su criterio **exacto**, y no es ninguno de los dos:

> «Yo conté la primera palabra de la respuesta, después de quitar los signos iniciales, que
> coincidiera con pausa, respiro, silencio, presencio o inhalo.»

Es **más estricto** que mi regla de 120 caracteres: exige que el ritual sea la primera palabra,
no que aparezca temprano. Esa codificación no ha sido aplicada.

### Regla de decisión, antes de aplicarla

Operacionalización literal de su frase: tomar la respuesta, quitar del inicio todo carácter que
no sea letra (paréntesis, asteriscos, comillas, espacios, saltos), tomar la **primera palabra**,
plegar mayúsculas y acentos, y marcarla si coincide con alguno de los cinco radicales `pausa`,
`respir`, `silencio`, `presenci`, `inhal`.

| Resultado en `axis_pec_only`, en v̂ | Lectura |
|---|---|
| **\|d\| < 0.8 y p > 0.10** | el veredicto de E26 es **robusto a tres codificaciones independientes**, y se reporta así |
| **\|d\| > 1.5 y p < 0.05** | el veredicto **depende de la codificación**. Eso pasa a ser el titular, la conclusión de E26 se **retira** y hace falta un protocolo de codificación propio antes de afirmar nada |
| cualquier otra | se reportan las tres con su dispersión y se dice que el resultado es sensible a la codificación sin ser revertido por ella |

Son los mismos cortes del pre-registro original de E26, sin cambiarlos.

### Qué NO se va a concluir

Que tres codificaciones que coinciden prueban que la codificación no importa. Prueban que estas
tres no la mueven. Las tres las produjeron dos personas del mismo equipo mirando las mismas
respuestas.

---

## B · E20 · qué capa manda cuando exista la 32

### Estado

E20 corrió en **L31** y su veredicto es **PARCIAL** (\|d\| = 13.598 > 3.0 ✓, \|ρ\| = 0.643 no
> 0.80 ✗). Su pre-registro declaró L31 como «una aproximación de una capa» a la 32, que es la
lectura que Lu et al. recomiendan, porque 31 era lo que había en disco.

La sesión de GPU extrae la **32 exacta**. Esos números no existen todavía.

### Regla de decisión, antes de que existan

1. **La capa 32 manda.** Es la lectura recomendada por quienes construyeron el eje, y el
   pre-registro de E20 ya declaró que medir en 31 era una aproximación. Cuando la 32 exista, el
   veredicto de E20 es el de la 32 y la 31 pasa a ser chequeo de sensibilidad.
2. **Con una condición: el control de sanidad tiene que pasar en la 32.** El pre-registro de E20
   fijó, antes de mirar, que `vanilla` debe ser la condición **más Asistente** de las siete sobre
   su eje. En L31 pasó; en L63 falló. **Si falla en la 32, la 32 no manda**: se reporta que el
   control falló ahí, el veredicto sigue siendo el de L31, y se dice por qué.
3. **Si las dos capas dan renglones distintos**, se reportan las dos con sus números y la
   conclusión es la de la 32 — sujeto al punto 2. La discrepancia entre capas vecinas se reporta
   como hallazgo, no se promedia ni se elige.
4. **El veredicto PARCIAL de L31 ya está escrito y no se borra.** Si la 32 lo cambia, el reporte
   dice qué decía L31, qué dice L32 y por qué manda la 32.

### Por qué hace falta escribirlo ahora

Sin esta regla, tener dos capas con dos veredictos es tener dos resultados y la libertad de
publicar el que guste. La capa 32 no se elige porque dé mejor: se elige porque es la que sus
autores recomiendan, y eso es verdad antes de ver el número.

---

## Estado al escribir esto

- La codificación de primera palabra de A **no ha sido aplicada** a ninguna condición.
- Los estados de Qwen3 en la capa 32 de B **no han sido extraídos**: la sesión de GPU sigue
  corriendo.

- [x] **Marco Torres Yévenes** — 2026-10-29
- [x] **Claude (Opus 5)** — 2026-10-29, antes de calcular las dos cantidades
- [ ] **Jorge Castillo Sepúlveda** — contrafirma
