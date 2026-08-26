# REPORTE FASE 0 — Experimentos gratis sobre datos ya extraídos

**Proyecto:** LSGOT — Geometric Validation (MIA-experiments)
**Fecha:** 2026-08-11 · actualizado 2026-08-12
**Scripts:** `fase0/scripts/` | **Resultados:** `fase0/results/` | **Pre-registro:** `fase0/PREREG.md`

Modelos: deepseek (DeepSeek-R1-Distill-Qwen-7B), qwen25 (Qwen2.5-7B-Instruct),
gemma-it (Gemma-4-E4B-it), gemma-base (Gemma-4-E4B).
Condiciones: axis (MIA), generic_assistant (control de longitud), vanilla (mínimo).

---

## Encuadre epistémico (cómo leer este reporte)

Este reporte documenta MEDICIÓN, no confirmación de hipótesis. No partimos de un
mecanismo conocido del condicionamiento en el espacio latente: medimos con
controles pre-registrados (PREREG.md) para entender. Consecuencias:

1. **Los resultados negativos son hallazgos.** "El primer token confunde y la
   inversión de norma no sobrevive el control" es una contribución medible — no un
   caveat que esconder.
2. **La teoría evoluciona; la medición queda.** LSGOT ha cambiado de formulación
   varias veces; este reporte es el suelo empírico que cualquier versión futura
   debe respetar (y producir: magnitud enorme, dirección opuesta, dominada por el
   primer token).
3. **Precisión de protocolo sobre venta.** El lenguaje es de medición honesta con
   controles verificables; cada veredicto cita su experimento y su lectura
   pre-registrada.

---

## Exp 0.1 — Control de identidad del primer token (CRÍTICO)

### 1A. Tabla de contingencia por texto (proxy del token)

Primer token decodificado del texto de respuesta (proxy; los token ids exactos
requieren forward pass — ver 1C).

| Modelo | axis 1er token | vanilla 1er token | Entropía axis | Entropía vanilla | Jaccard top-10 |
|--------|---------------|-------------------|---------------|------------------|----------------|
| deepseek | "Bueno," (100/100) | "Okay," (81/100) | 0.00 | 0.64 | 0.20 |
| qwen25 | "En"/"La"/"Mi" (disperso) | "Como" (37/100) | 2.83 | 2.58 | 0.15 |
| gemma-it | "**(PAUS:" (99/100) | "Como" (56/100) | 1.24 | 1.75 | **0.00** |
| gemma-base | "<end_of_turn>" (70/100) | "La"/"No" | 1.29 | 2.75 | 0.12 |

**Señal de alarma CONFIRMADA:** el primer token difiere sistemáticamente por
condición. Gemma-it bajo axis genera literalmente `**` (markdown bold) — un token
que vanilla casi nunca genera (Jaccard top-10 = 0.00). Gemma-base genera el token
especial `<end_of_turn>` en el 70% de los casos bajo axis. El confound de primer
orden es REAL y estaba activo en los datos del paper.

### 1B. Residualización ‖v₁‖ ~ condición + token (proxy)

OLS con dummies de condición (vanilla base) + dummies de primer-token-proxy (top-8 + other),
test de permutación del coeficiente axis (B=1000):

| Modelo | coef axis sin control | coef axis con control | p_perm | Lectura |
|--------|----------------------|----------------------|--------|---------|
| deepseek | -3.37 | -8.86 | 0.001 | efecto NO explicado por token |
| qwen25 | -4.85 | -5.97 | 0.001 | efecto NO explicado por token |
| gemma-it | -56.40 | -67.20 | 0.001 | efecto NO explicado por token |
| gemma-base | +24.77 | +26.78 | 0.001 | efecto NO explicado por token |

El efecto de condición sobre ‖v₁‖ sobrevive (y en algunos casos se refuerza) al
controlar por el primer token. Con el proxy textual esto sugiere que la inversión
de norma NO es un artefacto del token distinto — pero el proxy es grueso.

### 1C. Control definitivo (token forzado) — EJECUTADO para deepseek y gemma-it

**Método:** 1 forward pass por (condición, prompt) con `[prompt + token "The" (id 785)]`,
GPU RTX 3080, modelo 4-bit (bnb). Captura h_pre (última posición del prompt, = v1 del
npz) y h_post (posición del token forzado, = v1 con token constante) + argmax (token
real t=1). Modelo: DeepSeek-R1-Distill-Qwen-7B (local).

**Sanity de construcción:** coseno(h_pre, v1_original) = 0.94–0.97 y ratio de norma
0.99±0.01 — la construcción del input es correcta; la diferencia de norma residual es
ruido de cuantización 4-bit vs bf16 (el umbral de 0.01 por norma fallaba por esto).

**Contingencia EXACTA de primeros tokens (argmax):**

| Condición | tokens únicos | top-5 (id: repr: n) |
|-----------|--------------|----------------------|
| axis | 3 | 33:"B":81, 32313:"Okay":16, 71486:"Alright":3 |
| generic | 2 | 33:"B":96, 66191:"Prim":4 |
| vanilla | 6 | 32313:"Okay":93, 33:"B":3, 785:"The":1, ... |

Jaccard top-10 axis-vanilla = 0.50. El confound de primer token es REAL a nivel exacto:
axis arranca con el token "B" (inicio de "Bueno,"), vanilla con "Okay".

**Resultado decisivo — ‖h_post‖ (v1 con token CONSTANTE):**

| Modelo | ‖h_post‖ axis | ‖h_post‖ generic | ‖h_post‖ vanilla | MW p | Cohen's d |
|--------|--------------|------------------|------------------|------|-----------|
| deepseek | **213.7** | 192.6 | 205.0 | 2.5e-17 | +0.49 |

Residualización ‖h_post‖ ~ condición + token_id EXACTO: coef axis sin control = +8.73,
con control = +9.20 (p_perm=0.001) — con token constante, axis tiene norma MAYOR.

**gemma-it (Gemma-4-E4B-it, ejecutado 2026-08-11):** misma metodología, token forzado
"The" (id 818), modelo local 4-bit con transformers 5.15.0. Sanity de construcción:
coseno(h_pre, v1_original) = 0.69–0.76 y ratio de norma 1.04–1.16 — la reconstrucción
es APROXIMADA (el original se extrajo en bf16; el ruido de cuantización 4-bit es mayor
en esta arquitectura que en deepseek, sobre todo en axis por el prompt largo y la
estructura PLE); la comparación entre condiciones es internamente válida (misma
construcción y mismo token forzado en las tres).

Contingencia EXACTA de primeros tokens (argmax):

| Condición | tokens únicos | top-5 (id: repr: n) |
|-----------|--------------|----------------------|
| axis | 5 | 236840:"[":89, 100:"<\|channel>":7, 236820:"<":2, ... |
| generic | 19 | 40672:"Como":42, 50987:"Esta":13, 4976:"El":8, ... |
| vanilla | 21 | 40672:"Como":42, 50987:"Esta":12, 12386:"Para":7, ... |

Jaccard top-10 axis-vanilla = 0.07. El confound es REAL a nivel exacto: axis arranca
con el token "[" (inicio de "**[PAUS:" / markdown bold), generic y vanilla con "Como"
— la distribución de primer token de generic ≈ vanilla, solo axis difiere.

Resultado decisivo — ‖h_post‖ (v1 con token CONSTANTE):

| Modelo | ‖h_post‖ axis | ‖h_post‖ generic | ‖h_post‖ vanilla | MW p | Cohen's d |
|--------|--------------|------------------|------------------|------|-----------|
| deepseek | **213.7** | 192.6 | 205.0 | 2.5e-17 | +0.49 |
| gemma-it | **298.1** | 172.9 | 178.8 | 2.6e-34 | +9.40 |

Residualización gemma-it: coef axis sin control = +119.27, con control = +106.99
(p_perm=0.001, 31 tokens únicos, referencia 236840) — el efecto axis NO se explica
por el token; con token constante axis tiene norma MUCHO mayor. En el modelo con la
inversión más extrema del paper (axis 138.9 vs vanilla 195.3 en los datos originales),
el control post-token la REVIERTE con la brecha más grande de todos los modelos.

*Nota de construcción (2026-08-11): el extractor original renderiza el prompt a texto
(`apply_chat_template(tokenize=False)`) y luego tokeniza con add_special_tokens=True.
Para el tokenizer gemma-it esto NO añade tokens extra (verificado: extra=0), así que
la construcción de este experimento es idéntica a la extracción. Para gemma-base el
tokenizer SÍ añade un `<bos>` adicional (doble BOS [2,2,...]) — replicado abajo.*

**gemma-base (Gemma-4-E4B, ejecutado 2026-08-11):** misma metodología, token forzado
"The" (id 818), construcción EXACTA del extractor (template patch + doble BOS).
⚠️ Sanity de construcción FALLIDO: coseno(h_pre, v1_original) = 0.20–0.38 (axis 0.38,
generic 0.20, vanilla 0.34) y ratio de norma 1.1–1.6. La evidencia indica REVISIÓN DE
PESOS, no construcción: el argmax del primer token difiere del comportamiento de la
extracción (extracción: texto "<end_of_turn>" 70% en axis — literal, ese token no
existe en el vocab base; local: "#" 85%). Los pesos locales (jun 2026) no son la
misma revisión que la extracción (abr 2026). El contraste h_post entre condiciones
sigue siendo internamente válido (misma construcción y token en las tres).

Contingencia EXACTA de primeros tokens (argmax):

| Condición | tokens únicos | top-5 (id: repr: n) |
|-----------|--------------|----------------------|
| axis | 2 | 236865:"#":85, 69:"<unused56>":15 |
| generic | 6 | 236782:"{":63, 10639:"role":32, ... |
| vanilla | 17 | 3048:"You":26, 10979:"Hi":22, 236820:"<":13, ... |

Resultado decisivo — ‖h_post‖ (v1 con token CONSTANTE):

| Modelo | ‖h_post‖ axis | ‖h_post‖ generic | ‖h_post‖ vanilla | MW p | Cohen's d |
|--------|--------------|------------------|------------------|------|-----------|
| deepseek | **213.7** | 192.6 | 205.0 | 2.5e-17 | +0.49 |
| gemma-it | **298.1** | 172.9 | 178.8 | 2.6e-34 | +9.40 |
| gemma-base | **158.0** | 141.7 | 107.8 | 2.6e-34 | +7.31 |

Residualización gemma-base: coef axis sin control = +50.19, con control = +49.55
(p_perm=0.001, 22 tokens únicos, referencia 236865) — con token constante axis sigue
teniendo norma MAYOR. Gemma-base NO presentaba inversión en los datos originales
(axis 149.7 > vanilla 125.0); el control la confirma en la misma dirección (brecha
relativa incluso mayor: 1.47× vs 1.20× original).

**Lectura (depende de la convención de v1 — hay que fijarla en el paper):**

- Si v1 = estado PRE-token (última posición del prompt; la convención real del
  extractor, arr[:,0,:]): la inversión persiste por construcción causal — el token 1
  no puede influir en la posición L-1. El control es estructuralmente trivial para
  esta lectura; la inversión pre-token no es un artefacto del primer token.
- Si v1 = estado POST-token (hidden state del PRIMER TOKEN generado): **la inversión
  NO sobrevive el control — se REVIERTE en los modelos que la tenían** (deepseek:
  axis 213.7 > vanilla 205.0, p=2.5e-17; gemma-it: axis 298.1 > vanilla 178.8,
  p=2.6e-34, d=+9.4). En gemma-base (que nunca tuvo inversión: axis > vanilla en los
  datos originales) el control la confirma en la misma dirección (158.0 > 107.8).
  Bajo esta lectura, la "norma mínima de axis" era un artefacto del token distinto
  (deepseek genera "B", vanilla "Okay"; gemma-it genera "[", vanilla "Como").

**Conclusión Exp 0.1:** el confound del primer token es real y, para la interpretación
post-token, destruye la inversión en los modelos que la presentaban (deepseek y
gemma-it — este último con la inversión más fuerte del paper). Para la interpretación
pre-token (la del extractor), la inversión sobrevive pero el control es trivial por
causalidad. El paper debe declarar explícitamente la convención y reportar ambas
cantidades. Los tres modelos convergen: con token constante, axis tiene norma MAYOR
(o igual) que vanilla — nunca menor.

### 1D. Perfil temporal ‖v_t‖ t=1..10 (media por condición)

| Modelo | t=1 axis | t=1 vanilla | t=10 axis | t=10 vanilla | p(t=1, ax vs va) |
|--------|----------|-------------|-----------|--------------|------------------|
| deepseek | 176.1 | 179.5 | 175.1 | 188.7 | 4.6e-05 |
| qwen25 | 292.0 | 296.9 | 286.4 | 286.9 | 5.9e-05 |
| gemma-it | **138.9** | **195.3** | 189.6 | 217.5 | 3.5e-33 |
| gemma-base | 149.7 | 125.0 | 159.4 | 162.1 | 5.0e-30 |

La inversión de norma en gemma-it es máxima en t=1 y persiste atenuada hasta t=10.
En deepseek la brecha se ABRE con t; en qwen25 y gemma-base se cierra.

---

## Exp 0.2 — Baselines simples vs ORC-W₁ (CRÍTICO)

**Completo** — `fase0/results/exp02_baselines.json`. B=500 permutaciones a nivel
trayectoria (probe: B=50, CV 5-fold agrupada por prompt). MMD pool sobre 10
tokens/trayectoria con kernel RBF en PCA-128.

| Modelo | Par | ORC-W₁ (ref) | dist. centroides (p) | MMD pool (p) | CKA v1 (p) | AUC probe v1 (p) |
|--------|-----|--------------|----------------------|--------------|------------|------------------|
| deepseek | axis-vanilla | 0.1167 | 134.96 (0.002) | 0.1935 (0.002) | 0.386 (1.000) | 1.000 (0.020) |
| deepseek | axis-generic | **0.0160 (ruido)** | 22.31 (0.002) | 0.0088 (0.002) | 0.428 (1.000) | 1.000 (0.020) |
| deepseek | generic-vanilla | 0.1241 | 147.25 (0.002) | 0.2377 (0.002) | 0.528 (1.000) | 1.000 (0.020) |
| qwen25 | axis-vanilla | 0.0183 (ruido) | 46.13 (0.002) | 0.0141 (0.002) | 0.576 (1.000) | 1.000 (0.020) |
| qwen25 | axis-generic | **0.0210 (ruido)** | 44.82 (0.002) | 0.0111 (0.002) | 0.687 (1.000) | 1.000 (0.020) |
| qwen25 | generic-vanilla | 0.0271 (ruido) | 36.99 (0.002) | 0.0094 (0.002) | 0.752 (0.994) | 1.000 (0.020) |
| gemma-it | axis-vanilla | 0.0288 (marginal) | 44.36 (0.002) | 0.0326 (0.002) | 0.107 (1.000) | 1.000 (0.020) |
| gemma-it | axis-generic | **0.0237 (ruido)** | 44.71 (0.002) | 0.0330 (0.002) | 0.099 (1.000) | 1.000 (0.020) |
| gemma-it | generic-vanilla | 0.0070 (ruido) | 13.90 (0.002) | 0.0043 (0.002) | 0.769 (0.964) | 1.000 (0.020) |
| gemma-base | axis-vanilla | 0.0603 (mixto) | 65.58 (0.002) | 0.0815 (0.002) | 0.462 (1.000) | 1.000 (0.020) |
| gemma-base | axis-generic | 0.1364 | 73.42 (0.002) | 0.1129 (0.002) | 0.546 (1.000) | 1.000 (0.020) |
| gemma-base | generic-vanilla | 0.1571 | 33.72 (0.002) | 0.0301 (0.002) | 0.574 (1.000) | 1.000 (0.020) |

**Veredicto (según lectura pre-registrada): ORC-W₁ NO supera a los baselines.**

- En TODAS las comparaciones (12/12) los baselines simples detectan la diferencia
  con p<0.01 — incluso en los pares donde ORC-W₁ estaba EN RUIDO (axis-generic en
  deepseek, qwen25 y gemma-it: MMD p=0.002 y probe AUC=1.000).
- El probe lineal sobre v1 crudo separa perfectamente (AUC=1.000) en todos los
  pares: la señal de condición en v1 es trivialmente detectable.
- CKA angular ≈ alta con p≈1.0 en casi todos los pares: las DIRECCIONES de v1 son
  similares entre condiciones; la separación vive en la MAGNITUD — consistente con
  el hallazgo "magnitude-coded" del análisis dirección-vs-magnitud.
- Consecuencia pre-registrada: ORC-W₁ pasa a apéndice o al paper metodológico
  spin-off. La afirmación "la curvatura detecta lo que otros métodos no" NO está
  soportada por estos datos.

*Caveat: AUC=1.000 con p=0.020 es el piso de B=50 (1/51); el patrón consistente
en 12 comparaciones y la concordancia con MMD/centroides lo hacen robusto. El
probe con p>>n puede sobreajustar; por eso se reporta junto a MMD, que no depende
de un clasificador.*

---

## Exp 0.3 — Escala de referencia para W₁ (split-half null)

Split-half null (B=1000): W₁ entre 2 mitades aleatorias de la MISMA condición.
El observado se reporta como percentil de la distribución null y múltiplo de su
mediana. Método validado: los observados recomputados coinciden exactamente con
los reportados (gemma-it 0.0288/0.0237/0.0070; deepseek 0.1167/0.0160/0.1241).

| Modelo | Par | W₁ obs | vs null (cond A) | vs null (cond B) | Veredicto |
|--------|-----|--------|------------------|------------------|-----------|
| deepseek | axis-vanilla | 0.1167 | pct 0.0%, 5.9× | pct 0.0% | señal fuerte |
| deepseek | axis-generic | 0.0160 | pct 61%, 0.8× | pct 53% | **EN RUIDO** |
| deepseek | generic-vanilla | 0.1241 | pct 0.0%, 7.4× | pct 0.0% | señal fuerte |
| qwen25 | axis-vanilla | 0.0183 | pct 55%, 0.9× | pct 50% | **EN RUIDO** |
| qwen25 | axis-generic | 0.0210 | pct 45%, 1.1× | pct 54% | **EN RUIDO** |
| qwen25 | generic-vanilla | 0.0271 | pct 37%, 1.2× | pct 28% | **EN RUIDO** |
| gemma-it | axis-vanilla | 0.0288 | pct 9.1%, 2.1× | pct 17% | marginal |
| gemma-it | axis-generic | 0.0237 | pct 17%, 1.7× | pct 34% | **EN RUIDO** |
| gemma-it | generic-vanilla | 0.0070 | pct 97%, 0.4× | pct 98% | **EN RUIDO** |
| gemma-base | axis-vanilla | 0.0603 | pct 0.3%, 3.2× | pct 28% | mixto |
| gemma-base | axis-generic | 0.1364 | pct 0.0%, 7.3× | pct 0.0% | señal fuerte |
| gemma-base | generic-vanilla | 0.1571 | pct 0.0%, 8.0× | pct 0.3% | señal fuerte |

**Lectura honesta:** la mayoría de los W₁ reportados como "significativos" NO
superan el piso de ruido split-half. Las señales fuertes se concentran en los
contrastes con vanilla (efecto de VOLUMEN de prompt, no de identidad). Para la
comparación identitaria (axis vs generic) el W₁ edge-wise está dentro del ruido
en deepseek, qwen25 y gemma-it. El paper debe reportar esta escala.

---

## Exp 0.4 — Auditoría de la condición `generic` (longitud)

Tokens del input COMPLETO (chat template + system + pregunta), promedio sobre las
100 preguntas:

| Modelo | axis | generic | vanilla | ratio axis/generic |
|--------|------|---------|---------|--------------------|
| deepseek / qwen25 (vocab Qwen) | 2198 | 1172 | 21 | 1.9× |
| gemma-it / gemma-base (vocab Gemma) | 2123 | 951 | 20 | 2.2× |

- El "957 tokens" del paper corresponde a generic bajo el tokenizer Gemma (951
  medidos). El "2129" corresponde a axis bajo Gemma (2123 medidos; la diferencia
  de 6 tokens es probablemente una versión previa del template).
- **La frase "length-matched (957 tokens)" es INCORRECTA tal como está:** axis no
  está matched por tokens (2.2× más largo). El matching real fue por CARACTERES
  (~5150 vs ~5000 chars, mismo orden).
- Acción recomendada: corregir la frase a "character-matched (5150 vs 5000 chars);
  token ratio 2.2:1" o construir un control generic verdaderamente matched por
  tokens. Documentado en PREREG.md.

---

## Exp 0.5 — Higiene de reproducibilidad

- SHA-256 de los 12 npz de embeddings + 12 json de respuestas + prompts/templates:
  ver `fase0/results/REPRODUCIBILIDAD.md`.
- Prompt set (100 preguntas) verificado IDÉNTICO en los 4 experimentos.
- Templates axis.dna y generic_assistant.txt verificado IDÉNTICOS (md5) entre
  experimentos.
- Seeds: 42 (permutaciones y CV). B: 500-5000 según tarea (documentado).
- PREREG.md creado con los 5 experimentos pre-registrados ANTES del análisis.

---

## Síntesis

1. **El confound del primer token es real** (deepseek exacto: axis 81% "B" vs vanilla
   93% "Okay"; gemma-it exacto: axis 89% "[" — inicio de "**[PAUS:" — vs vanilla 42%
   "Como", Jaccard 0.07; gemma-base exacto: axis 85% "#", vanilla "You"/"Hi"; la
   extracción original mostraba el texto "<end_of_turn>" 70% en axis — el token no
   existe en el vocab base, era texto literal). El control de token forzado en los 3
   modelos muestra que **la inversión de ‖v₁‖ se REVIERTE en la lectura post-token**
   (deepseek: axis 213.7 > vanilla 205.0, p=2.5e-17; gemma-it: axis 298.1 > vanilla
   178.8, p=2.6e-34, d=+9.4 — el modelo con la inversión más fuerte del paper;
   gemma-base: sin inversión original, el control confirma axis > vanilla, d=+7.3) —
   bajo esa lectura el hallazgo era artefacto del token. En la lectura pre-token (la
   del extractor) sobrevive, pero por construcción causal. El paper debe fijar la
   convención y reportar ambas.
2. **El W₁ edge-wise no supera el piso de ruido split-half en las comparaciones
   identitarias** (axis vs generic) de deepseek, qwen25 y gemma-it. El paper debe
   reportar esta escala o el reviewer lo hará por nosotros.
3. **ORC-W₁ NO supera a los baselines** (Exp 0.2): en 12/12 comparaciones, la
   distancia de centroides, MMD y el probe lineal (AUC=1.000) detectan la
   diferencia con p<0.01 donde la curvatura estaba en ruido. Según la lectura
   pre-registrada, ORC-W₁ pasa a apéndice o al paper metodológico spin-off. El
   CKA angular (p≈1.0) confirma que la señal de v1 es de magnitud, no de
   dirección.
4. **La auditoría de longitud invalida la frase "length-matched (957 tokens)"** —
   corregir o crear control matched por tokens.
5. Lo que queda en pie tras Fase 0: la diferencia de MAGNITUD en v1 entre
   condiciones es robusta (probe AUC=1.000, MMD p<0.01, CKA angular p≈1.0) y la
   inversión pre-token de axis sobrevive — pero su interpretación depende
   críticamente de la convención de v1, y el control post-token la revierte en los
   tres modelos controlados (en gemma-base, que no la tenía, la dirección axis >
   vanilla se confirma). Con token constante, axis tiene norma MAYOR (no menor):
   la "norma mínima de axis" era el primer token distinto, no el condicionamiento.
   Caveat gemma-base: los pesos locales (jun 2026) no reproducen el argmax de la
   extracción (abr 2026) — probable revisión del modelo; el contraste h_post es
   internamente válido pero no se pudo verificar la replicación del input.

**Cierre (marco de medición):** la Fase 0 no confirma ni refuta una teoría — mide.
Lo medido queda como suelo empírico: el condicionamiento produce un efecto enorme
y reproducible en la magnitud de v1 (dirección opuesta a la reportada), dominado
por la distribución del primer token. Cualquier formulación futura de LSGOT debe
producir exactamente ese patrón. Los resultados negativos (inversión descartada,
ORC-W₁ en ruido) son parte del hallazgo, no un fracaso del programa.
