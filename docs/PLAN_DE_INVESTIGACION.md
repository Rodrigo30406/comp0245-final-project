# Plan de investigación — COMP0245: Behaviour Cloning para un manipulador móvil

**Autoría del proyecto:** Rodrigo y equipo. **Fecha del plan y consulta bibliográfica:** 16 de octubre de 2026. **Estado:** propuesta de investigación; experimentos pendientes. **Rama de trabajo:** `rodrigo`.

Este documento define un programa amplio de investigación. El alcance se podrá ajustar después, conservando las preguntas y los controles necesarios para interpretar los resultados. Las fases se ordenan por dependencias científicas, sin imponer todavía un calendario de implementación.

**Objetivo:** entrenar un regresor de aprendizaje automático mediante *behaviour cloning* (BC) para producir velocidades de las ruedas y torques del brazo a partir del estado del robot y una posición objetivo; medir su precisión, generalización, robustez y eficiencia. La pregunta central es **qué combinación de cobertura de datos, información observable y memoria temporal permite imitar al experto y alcanzar objetivos nuevos de forma fiable, y con qué coste computacional**.

Las cifras de datasets, arquitecturas, perturbaciones y umbrales experimentales que se proponen aquí son **decisiones iniciales nuestras**, salvo cuando se atribuyen expresamente a las diapositivas o al código. Deben calibrarse en el piloto y congelarse antes de los experimentos confirmatorios. Ningún resultado esperado debe presentarse como resultado obtenido.

**Guía de lectura:** las secciones 1–5 establecen el problema y las hipótesis; 6–9 diseñan datos y modelos; 10–13 definen cómo juzgar los resultados; 14–19 organizan ejecución, informe y referencias.

<details>
<summary>Índice del plan</summary>

- [1. Assignment y calificación](#1-contrato-del-assignment-y-trazabilidad-con-la-calificación)
- [2. Formulación y experto](#2-formulación-matemática-y-significado-de-experto)
- [3. Auditoría del código](#3-auditoría-del-código-que-condiciona-los-experimentos)
- [4. Fundamento bibliográfico](#4-bibliografía-fundamento-actualidad-y-límites-de-transferencia)
- [5. Hipótesis](#5-preguntas-hipótesis-y-posibles-refutaciones)
- [6. Diseño de datos](#6-metodología-de-datos-cantidad-calidad-y-cobertura)
- [7. Entradas y sensores](#7-entradas-y-sensores-qué-existe-y-qué-debemos-validar)
- [8. Tiempo y Fourier](#8-un-instante-ventanas-temporales-y-fourier)
- [9. Arquitecturas y entrenamiento](#9-arquitecturas-del-regresor-y-entrenamiento)
- [10. Precisión y generalización](#10-evaluación-precisión-y-generalización)
- [11. Robustez](#11-robustez-y-aprendizaje-de-recuperación)
- [12. Eficiencia](#12-eficiencia-y-frontera-de-desempeño)
- [13. Matriz de experimentos](#13-matriz-de-experimentos-y-controles)
- [14. Reproducibilidad](#14-ingeniería-experimental-y-reproducibilidad)
- [15. Fases](#15-fases-de-ejecución-y-criterios-para-avanzar)
- [16. Informe](#16-figuras-tablas-y-relato-científico-del-informe)
- [17. Cuestiones pendientes](#17-incertidumbres-y-decisiones-pendientes)
- [18. Referencias](#18-bibliografía-anotada-y-fuentes)
- [19. Glosario](#19-glosario-para-discutir-el-proyecto)

</details>

## 1. Contrato del assignment y trazabilidad con la calificación

Fuente principal: [diapositivas COMP0245 Assignment 2026/27](COMP0245-Assignment-2026-2.pdf). Los números siguientes cuentan la portada como diapositiva 1.

| Diapositivas | Qué establece el profesor | Evidencia que produciremos |
|---|---|---|
| 2–4: objetivo y outline | BC supervisado a partir de trayectorias expertas; reproducir la tarea y generalizar a objetivos y/o condiciones iniciales nuevos | Dataset de pares observación–acción; regresor; ejecuciones autónomas con objetivos y condiciones reservados |
| 4: interfaz | Entradas base: `base_pos`, `base_bearing`, `q_mes`, `qd_mes`, `TARGET_EE_POS`; salida: `wheel_cmds`, `torque_arm`; se permite explorar información extra | Baseline explícito con esas entradas; ablaciones que añaden información sin cambiar silenciosamente la tarea |
| 4–5: metodología | Generar datos; diseñar evaluación; justificar cantidad/calidad, entradas, instante frente a intervalo temporal, arquitectura, entrenamiento e hiperparámetros | Protocolo de muestreo, curvas de aprendizaje, auditoría del dataset, comparaciones controladas y registro de decisiones |
| 6: precisión | Evaluar cuán precisa es la solución | Error de acciones y métricas de llegada al objetivo, distinguiendo evaluación offline y control en simulación |
| 6: generalización | Evaluar otros objetivos y condiciones iniciales | Pruebas separadas de interpolación, regiones reservadas, extrapolación y nuevas configuraciones iniciales |
| 6: robustez | Evaluar adversidades | Curvas de degradación ante ruido, retraso y perturbaciones físicas definidas |
| 6: eficiencia | Memoria, número de parámetros y tiempo de cómputo | Memoria de inferencia, parámetros, latencias y cumplimiento del periodo de control |
| 7: hitos | Briefing 20/10; demostración presencial entre 30/10 y 06/11, obligatoria aunque no puntuada; entrega 06/11 | Demo reproducible; informe y código. Las fechas son las indicadas en este PDF |
| 8–10: entregables | Informe IEEE LaTeX de máximo 6 páginas; contribuciones de máximo 1 página, posible página 7; README con ejecución, outputs, fuentes y modificaciones | Informe coherente con los experimentos; instrucciones verificadas; atribución del código; declaración de contribuciones |
| 10: archivos de entrega | No es necesario incluir datos ni pesos/checkpoints en Moodle; se permite enlace a GitHub si el código excede el límite | Procedimiento de regeneración y manifiestos; acceso a artefactos cuando corresponda |
| 11–14: calificación | Corrección metodológica, comprensión, reproducibilidad, justificación e interpretación; ideas novedosas bien utilizadas para el tramo superior | Resultados repetidos, análisis de fallos y una contribución experimental defendible |

La rúbrica describe 0–49 como trabajo ampliamente incompleto o con fallos fundamentales; 50–59 con problemas metodológicos importantes, uso poco comprendido de recursos o escasa justificación; 60–69 con metodología generalmente correcta pero limitaciones técnicas o de reproducibilidad; 70–79 con errores muy menores, documentación y comprensión excelentes; 80–100 añade ideas novedosas bien justificadas en el método o la validación. **El número de modelos probados no constituye por sí mismo una aportación.**

La formulación solicitada es regresión supervisada de acciones continuas. El tronco del proyecto será BC. DAgger se estudia como extensión de imitación interactiva y los modelos generativos como extensiones de BC; deben identificarse como tales y mantener una comparación limpia con el regresor base. El aprendizaje por refuerzo, los modelos de lenguaje y la manipulación de objetos no son requisitos de estas diapositivas.

## 2. Formulación matemática y significado de “experto”

El experto es el controlador programado del profesor: navegación de la base por fases y control del brazo mediante cinemática inversa diferencial, PD y compensación gravitatoria. En una trayectoria, observa el robot y emite acciones en cada paso.

Definimos la observación mínima según el esquema de las diapositivas:

$$o_t = [p^b_t,\ \psi_t,\ q^a_t,\ \dot q^a_t,\ g],$$

donde $p^b_t\in\mathbb{R}^3$ es la posición de la base, $\psi_t$ su orientación planar, $q^a_t,\dot q^a_t\in\mathbb{R}^7$ las posiciones y velocidades del brazo y $g\in\mathbb{R}^3$ el objetivo cartesiano. Son **21 escalares**; representar el bearing con seno y coseno da 22.

La salida requerida es:

$$a_t = [\omega^{FR}_t,\omega^{FL}_t,\omega^{BL}_t,\omega^{BR}_t,\tau^1_t,\ldots,\tau^7_t]\in\mathbb{R}^{11}.$$

Las ruedas reciben velocidades en rad/s y el brazo torques en N·m. Los dedos se mantienen en su configuración fija, como en el código base. **Predecir velocidades articulares del brazo o posiciones deseadas sería cambiar la interfaz de salida**; cualquier estudio de esa variante debe quedar separado del resultado principal.

```mermaid
flowchart LR
    S[Escenarios de entrenamiento] --> E[Experto y simulador]
    E --> D[Episodios con observaciones y acciones]
    D --> T[Entrenamiento supervisado del regresor]
    T --> P[Política aprendida]
    G[Escenarios reservados] --> V[Simulador de evaluación]
    V -->|Observación actual e historial permitido| P
    P -->|4 velocidades de ruedas y 7 torques| V
    V --> M[Precisión, generalización, robustez y coste]
```

El dataset contiene episodios completos $D=\{\tau_i\}$ y pares $(o_t,a_t^E)$ del experto. Entrenamos $\hat a_t=f_\theta(o_t)$ o $\hat a_t=f_\theta(o_{t-H+1:t})$ con una pérdida supervisada. Durante la evaluación autónoma las acciones del modelo cambian el estado siguiente del robot; por eso un error pequeño en el dataset puede acumularse durante la ejecución. Esta dependencia secuencial motiva la evaluación en lazo cerrado y la posible ampliación con DAgger. [Ross et al., 2011](https://proceedings.mlr.press/v15/ross11a.html).

**La memoria del experto importa.** Su acción depende también de un estado interno $h^E_t$: fase de navegación, referencia acumulada del brazo, postura inicial mantenida y tiempo de ejecución. Formalmente, $a_t^E=\pi_E(s_t,g,h_t^E)$. La observación mínima puede no identificar toda esa información. Que dos observaciones parecidas tengan etiquetas diferentes podría indicar observabilidad insuficiente, además de posibles errores de registro. Una ventana corta tampoco garantiza recuperar toda la historia.

## 3. Auditoría del código que condiciona los experimentos

Versiones inspeccionadas: starter del profesor `cae4fb69386872ae73c601ce77e182aad7abe284`; proyecto local `c8c1bfb42d5d3c6c1760df4543461685ba8bf6fb`; RoboEnv `7bb49439677c2ac111347b40e997025ce87bff50`; submódulo `simulation_and_control` `df5850526affcc33b25b15d3096c4d21b614bacd`. La inspección ha sido estática, sin validar todavía una simulación. El checkout de RoboEnv muestra contenido no versionado dentro de su submódulo; los hashes por sí solos no bastarán para reproducir ese entorno.

Fuentes: [bucle de la tarea](../final/robotnik_arm_task/robotnik_arm_task.py), [controladores](../final/robotnik_arm_task/robotnik_arm_control.py), [instrucciones originales](../final/robotnik_arm_task/README.md). Para inspecciones locales: [interfaz PyBullet de RoboEnv](../../RoboEnv/simulation_and_control/simulation_and_control/sim/pybullet_robot_interface.py) y [configuración Robotnik](../../RoboEnv/configs/robotnik_arm.json). Estos dos últimos enlaces requieren la estructura local de carpetas; para el equipo se registrarán también repositorio, commit y cambios del entorno.

| Hallazgo | Consecuencia metodológica |
|---|---|
| `q_mes` y `qd_mes` del script contienen 17 articulaciones: 4 ruedas, 7 del brazo y 6 dedos; el brazo es `[4:11]` | Guardar arrays completos con nombres, pero definir el baseline de 21 variables con las 7 articulaciones del brazo. La interpretación de las diapositivas y las dimensiones reales deben quedar explícitas |
| `base_controller` mantiene las fases `align`, `drive`, `orient` y un indicador `done` separado | Medir errores por fase; guardar fase para análisis. No entregarla gratuitamente a una política que pretendemos autónoma |
| `arm_controller` integra `q_des_arm` y mantiene `q_init_arm` mientras navega | Comparar memoria y acceso a la postura inicial; no usar referencias expertas como entradas normales |
| La orientación deseada de la base se calcula desde `atan2(target_y, target_x)`, referida al origen del mundo | No asumir que trasladar o rotar un episodio produce etiquetas equivalentes. Auditar las condiciones iniciales desplazadas y comparar representaciones absolutas y relativas |
| El experto tiene timeout de navegación de 25 s y límite de alcance del brazo de 0.85 m | Un fallo del experto no demuestra imposibilidad física; podría ser una limitación de la estrategia o del tiempo disponible |
| Tolerancia de llegada configurada en 0.01 m, mínimo temporal de 8 s y parada por estancamiento | Registrar el criterio original; definir además evaluación externa común que no dependa de la fase interna del experto |
| Configuración local con `time_step=0.001` y máximo por defecto de 40 000 pasos | Frecuencia física nominal de 1 kHz y horizonte aproximado de 40 s. Confirmar el tiempo real del bucle; distinguir frecuencia física, de control, de muestreo y de historial |
| El logging actual sirve para plots, no guarda directamente un dataset completo de acciones expertas | Registrar observación y acción antes de `sim.Step`; registrar estado siguiente después. Guardar también el último paso, actualmente susceptible de quedar fuera al romper el bucle |
| Las mediciones usadas en la comprobación de parada preceden al `sim.Step` de esa iteración | Alinear explícitamente timestamps y estados terminales para evitar desfase de un paso en métricas |
| Los comandos del experto emparejan las ruedas derechas y las izquierdas | Ablación con dos velocidades independientes expandidas a cuatro salidas, frente a cuatro salidas libres |
| `GetMotorTorques` devuelve `applied_motor_commands` | Es un comando aplicado por la capa de actuadores, no evidencia de un sensor físico independiente de torque |
| Se almacenan `motor_accelerations` y la clave plural del buffer, pero `GetMotorAccelerationTMinusOne` lee nombres en singular | Inconsistencia visible por inspección; validar/corregir con trazabilidad o calcular diferencias causales fuera del getter antes de usar aceleración |
| Algunos getters generan ruido en cada llamada; campos `*_cov` se pasan como `scale` de `np.random.normal` | El valor se usa como desviación estándar en esas llamadas. Centralizar observaciones por paso y comprobar unidades; evitar añadir ruido distinto a cada lectura de la misma señal |
| El getter de velocidad lineal en marco corporal contiene `self.self.bot` en la rama de ruido | Validar ese camino antes de usarlo; una rotación explícita de velocidades del mundo es una alternativa verificable |
| `collision_enable=0` desactiva la opción de autocollision del URDF en esta implementación | No interpretar ausencia de contactos registrados como ausencia de auto-intersección. Auditar geometría antes de afirmar seguridad de movimiento |

El controlador de ruedas continúa convirtiendo velocidades solicitadas en acciones de los actuadores. Mantener esta capa común para experto y modelo forma parte de la interfaz proporcionada; sustituir los dos controladores de alto nivel no elimina toda la física o el control de bajo nivel.

**Alcance del repositorio:** se excluyen `week1/` y `week2/`, ejercicios independientes de laboratorio que el código final no importa. El starter de `final/robotnik_arm_task/` depende de su controlador local y de bibliotecas externas, incluida `simulation_and_control` de RoboEnv. Los laboratorios siguen disponibles en upstream y en el historial; su eliminación no cambia el experto de la tarea final.

## 4. Bibliografía: fundamento, actualidad y límites de transferencia

La revisión combina fundamentos con trabajos recientes disponibles hasta la fecha de consulta. Es una selección dirigida a las preguntas del assignment, no una revisión sistemática exhaustiva ni una afirmación de tener el modelo más reciente de todo el campo. Las mejoras publicadas en otras tareas no son mejoras demostradas en este robot.

| Trabajo | Aporte que guía este plan | Límite de la extrapolación |
|---|---|---|
| [Ross et al., DAgger, 2011](https://proceedings.mlr.press/v15/ross11a.html) | Desplazamiento de distribución provocado por las acciones de la política; recolección iterativa de correcciones | Nuestro experto tiene memoria y debe definirse cuidadosamente cómo consultarlo fuera de sus trayectorias |
| [Mandlekar et al., CoRL 2021 / actas 2022](https://proceedings.mlr.press/v164/mandlekar22a.html) | BC recurrente, calidad de demostraciones, selección de observaciones y dificultad de seleccionar checkpoints por pérdida offline | Sus demostraciones humanas y tareas de manipulación no determinan cuántas trayectorias necesita Robotnik |
| [Lin et al., 2024; revisión v4 de junio de 2026](https://arxiv.org/abs/2410.18647v4) | Estudia diversidad y volumen de demostraciones; motiva separar cobertura de condiciones y repetición | Diversidad visual de objetos/entornos es distinta de diversidad de objetivos y estados; esta aplicación es una hipótesis nuestra |
| [Dass et al., DataMIL, 2025; revisión v2 de junio de 2026](https://arxiv.org/abs/2505.09603v2) | Seleccionar datos por su utilidad para el desempeño puede ser mejor que acumular datos indiscriminadamente | Sus mezclas de grandes datasets son distintas de nuestros datos generados; usaremos primero controles de selección sencillos |
| [Laskey et al., DART, 2017](https://proceedings.mlr.press/v78/laskey17a.html) | Demostraciones con perturbaciones y recuperación para ampliar cobertura | Añadir ruido arbitrario no equivale a implementar DART completo |
| [Bai et al., 2018](https://arxiv.org/abs/1803.01271) | Convoluciones temporales como alternativa a recurrencia | Comparación general de secuencias, no evidencia directa sobre este controlador |
| [Zhao et al., ACT, 2023](https://arxiv.org/abs/2304.13705) | Predicción de secuencias de acciones mediante transformers | Action chunks de otras interfaces no garantizan estabilidad al aplicar torques de brazo en lazo abierto |
| [Chi et al., Diffusion Policy, 2023/2024](https://arxiv.org/abs/2303.04137) | BC generativo de acciones y control con horizonte recedente | La capacidad de representar multimodalidad puede no compensar su coste en un experto determinista de baja dimensión |
| [Wang et al., OneDP, ICML 2025](https://proceedings.mlr.press/v267/wang25ba.html) | Destilación de una política de difusión para reducir los pasos de generación | Acelerar el modelo generativo no garantiza cumplir nuestro periodo de control |
| [Pertsch et al., FAST, 2025](https://arxiv.org/abs/2501.09747) | Representación de secuencias de acciones mediante transformada coseno | FAST tokeniza acciones para VLAs; no demuestra que una FFT de nuestras observaciones mejore la regresión |
| [Tancik et al., NeurIPS 2020](https://arxiv.org/abs/2006.10739) | Fourier features para regresión de funciones de coordenadas | Codificación de entradas y análisis espectral temporal son intervenciones distintas |
| [Agarwal et al., NeurIPS 2021](https://arxiv.org/abs/2108.13264) | Importancia de incertidumbre y agregación cuidadosa al comparar algoritmos | Adaptamos principios estadísticos de evaluación, sin asumir que este proyecto sea RL |
| [Shi et al., FODMP, preprint de marzo de 2026](https://arxiv.org/abs/2603.24806) | Línea reciente sobre generación rápida y estructura temporal de movimientos | Referencia exploratoria; aquí se consultó su resumen. No es una implementación propuesta como baseline ni un resultado validado en Robotnik |

Una contribución plausible del proyecto sería demostrar, con controles, cuándo una memoria breve o velocidades adicionales sustituyen parte de la información que falta en una observación instantánea, y cómo cambia ese efecto al ampliar la cobertura de demostraciones. La novedad sería contextual y experimental; no se afirmará inventar un algoritmo globalmente nuevo.

## 5. Preguntas, hipótesis y posibles refutaciones

| ID | Hipótesis a contrastar | Comparación principal | Qué debilitaría la hipótesis |
|---|---|---|---|
| H1 | A presupuesto de datos comparable, cubrir más objetivos y estados iniciales mejora la generalización más que repetir pocas condiciones | Diversidad alta frente a repetición alta; igual número de episodios y análisis adicional con igual número de transiciones | Diferencias pequeñas o invertidas al controlar arquitectura, duración y semillas |
| H2 | Velocidades de base y ruedas aportan información dinámica útil, especialmente con deslizamiento | Entradas base frente a base + velocidades | No mejora el control autónomo, o el beneficio desaparece con ventanas temporales |
| H3 | La memoria ayuda a imitar un experto con fases y referencias internas | MLP instantánea, MLP con ventana, TCN y GRU; presupuesto de parámetros comparable | No hay mejora reproducible o solo aparece al dar más capacidad al modelo temporal |
| H4 | Aceleraciones y acciones pasadas ayudan bajo algunas condiciones, pero pueden aumentar sensibilidad al ruido | Añadir cada grupo por separado y cruzarlo con ruido/retraso | Ganancia solo con señales ideales o cuando se usa accidentalmente información futura |
| H5 | Características espectrales causales pueden resumir dinámica útil con menor coste que una ventana completa | Ventana cruda, resúmenes temporales simples y FFT/DCT con igual historial | No superan resúmenes simples, pierden información de fase o empeoran transiciones |
| H6 | Modelos compactos pueden aproximarse al éxito de modelos grandes con menor latencia y memoria | Curvas capacidad–éxito–coste y comparación de arquitecturas | La reducción de coste implica pérdidas relevantes de éxito o robustez |
| H7 | Demostraciones de recuperación reducen fallos derivados de desviarse del dataset | BC nominal frente a BC con recuperación; DAgger como extensión | La ganancia se explica solo por más muestras o por ayuda experta durante la evaluación |

Cada experimento registrará hipótesis, intervención, variables controladas, métrica primaria, criterio de interpretación y fallos posibles **antes de mirar el test final**. Las hipótesis pueden rechazarse; un resultado negativo bien explicado es útil.

## 6. Metodología de datos: cantidad, calidad y cobertura

### 6.1 Piloto del experto y definición del dominio

Primero validar la ejecución nominal con el objetivo original `[0.0, 1.1, 0.5]`, sin alterar ganancias, física ni estrategia. Registrar el entorno realmente utilizado y verificar orden de articulaciones, signos, unidades, límites y timestamps.

Después proponer un piloto de **64 condiciones**: 8 sectores angulares × 4 radios × 2 alturas, con postura nominal. Como dominio inicial de exploración, considerar radio horizontal respecto al origen entre 0.6 y 1.4 m y altura entre 0.3 y 0.8 m. Estos rangos no son un workspace demostrado: deben someterse al experto y a comprobaciones geométricas. Incluir delante, detrás y ambos lados; no concentrar todas las demostraciones cerca del objetivo original.

Ampliar el piloto con variaciones de estado inicial: desplazamiento planar de hasta ±0.15 m, yaw hasta ±15°, articulaciones del brazo hasta ±0.1 rad respecto a la postura nominal y velocidades iniciales cero. Comprobar límites, ausencia de interpenetración y transitorio inicial. Más adelante variar también velocidades iniciales y perturbaciones mayores. Registrar como condiciones distintas las combinaciones de objetivo y postura; el piloto se utiliza para desarrollo, no como test final.

Clasificar cada ejecución en: llegada, timeout, estancamiento, límite articular, contacto no permitido, inestabilidad numérica u otro fallo identificado. **“El experto falló” y “el objetivo es físicamente inalcanzable” son afirmaciones diferentes.** La función original puede imprimir que el objetivo no es alcanzable por observar estancamiento; ese mensaje no es una prueba geométrica.

Congelar dos conjuntos de evaluación: uno de condiciones calificadas en las que el experto alcanza el objetivo y otro de estrés más amplio. Conservar los fallos y la tasa de rechazo de la calificación para que el dominio declarado sea transparente. Nunca quitar un caso del test porque el modelo falle.

### 6.2 Diseño de muestreo

Representar cada escenario por `scenario_id` y un vector de factores: objetivo, pose inicial de base, postura/velocidad inicial del brazo, parámetros físicos y semilla de perturbación. Guardar también la medida de muestreo: uniforme en radio no equivale a uniforme en área. Si se busca uniformidad en un anillo, muestrear $r=\sqrt{u(r_{max}^2-r_{min}^2)+r_{min}^2}$, con $u$ uniforme, y ángulo uniforme.

Comparar muestreo aleatorio con estratificación por sector, radio y altura; usar Latin hypercube o Sobol como candidatos para cubrir factores continuos. No afirmar cobertura del espacio completo solo por usar esos métodos: mostrar histogramas, celdas ocupadas y distancias a vecinos del conjunto de entrenamiento. Cualquier distancia se define con escalas físicas o normalización de entrenamiento, no con unidades mezcladas sin criterio.

Para H1, construir condiciones con igual presupuesto: por ejemplo, 512 episodios repartidos entre 512 objetivos/poses diversos frente a 64 condiciones con 8 ejecuciones cada una. En un simulador determinista, repetir exactamente una condición puede producir duplicados: identificarlo y reportarlo. Para estudiar repetición con variabilidad, cambiar de manera explícita pequeñas perturbaciones o semillas, manteniendo los mismos centros de las condiciones.

Separar dos comparaciones: mismo número de episodios y mismo número de transiciones utilizadas en entrenamiento. Las trayectorias largas no deben ganar automáticamente más peso ni confundirse con mayor diversidad. Añadir un análisis a presupuesto de cómputo comparable para evitar atribuir al dataset una ganancia debida a más pasos de optimización.

### 6.3 Cuánto dato: una decisión basada en evidencia

Usar conjuntos de entrenamiento anidados de **32, 64, 128, 256, 512, 1 024, 2 048 y 4 096 episodios**, ampliables si sigue habiendo mejora relevante. Son puntos propuestos para una curva, no una afirmación de que 4 096 sea necesario. Separar de estos episodios la validación y las pruebas. Construir al menos tres órdenes de selección estratificada para comprobar que una curva favorable no dependa de un subconjunto afortunado.

Para cada tamaño medir éxito en validación, error cartesiano, cobertura, errores por subgrupo y dispersión entre semillas. Comparar inicialmente una MLP y un modelo temporal para no declarar “datos suficientes” solo porque una arquitectura limitada haya llegado a su techo. Ajustar una curva de saturación o ley de potencia únicamente si los datos y residuos lo respaldan; no extrapolarla como garantía.

**Criterio operativo de suficiencia propuesto:** tras dos duplicaciones, la mejora estimada de éxito es menor de 2 puntos porcentuales y el intervalo de confianza de la mejora es suficientemente estrecho para descartar una ganancia relevante; no persisten huecos importantes de cobertura ni deterioro de grupos difíciles. Dos puntos es una decisión práctica a registrar antes del análisis. Si el intervalo es amplio, el resultado es inconcluso: ampliar la evaluación o revisar el diseño. Considerar también la reducción del error final, aunque el éxito binario se haya estabilizado.

La justificación bibliográfica es metodológica. Mandlekar et al. estudian fracciones del dataset y muestran dependencia de la dificultad de la tarea. Lin et al. separan diversidad y repetición, con rendimientos decrecientes en sus tareas. Eso respalda **medir nuestra curva**, no copiar su número de demostraciones. [Mandlekar et al.](https://proceedings.mlr.press/v164/mandlekar22a.html), [Lin et al.](https://arxiv.org/abs/2410.18647v4).

Reportar siempre: intentos de recolección, episodios válidos, episodios exitosos, condiciones distintas, segundos simulados, transiciones guardadas, transiciones muestreadas para entrenar y espacio en disco. Una trayectoria de 40 s a 1 kHz tiene aproximadamente 40 000 pasos, pero no 40 000 ejemplos independientes de comportamiento.

### 6.4 Calidad y selección

Aplicar primero controles de integridad: valores finitos, timestamps monótonos, longitudes alineadas, límites conocidos, nombres de articulaciones, acciones aplicadas y motivo de terminación. Guardar los datos crudos y un manifiesto de aceptación/rechazo; no sobrescribirlos con versiones filtradas.

La calidad se describirá mediante varias variables: éxito, error final, tiempo, saturación de comandos, transiciones de fase, contactos, estabilidad y cobertura. Una demostración lenta puede contener recuperaciones valiosas. Una demostración rápida no es necesariamente mejor para enseñar estabilidad.

Comparar cuatro composiciones, controlando tamaño y cobertura:

| Dataset | Composición | Pregunta |
|---|---|---|
| Q0 | Episodios válidos y exitosos del experto nominal | Referencia limpia |
| Q1 | Q0 con equilibrio entre episodios/fases, o ponderaciones explícitas | ¿Evitar que dominen largos tramos repetitivos ayuda? |
| Q2 | Q0 más recuperaciones exitosas tras perturbaciones durante la demostración | ¿Se aprende a corregir desviaciones? |
| Q3 | Mezcla controlada con episodios válidos pero fallidos o subóptimos; proporciones propuestas 10%, 25%, 50% | ¿Cuándo datos adicionales ayudan y cuándo degradan BC? |

El balance por fase solo usa metadatos de entrenamiento; reportar también pérdida con la distribución natural y su desempeño autónomo. Registrar si se usan pesos inversos para corregir el muestreo o si se cambia deliberadamente el objetivo de entrenamiento. Las ventanas temporales conservan su continuidad aunque su punto final se muestree de manera estratificada.

La selección por incertidumbre, cobertura o fallo de validación será una extensión con un conjunto de validación propio. Comparar con selección aleatoria del mismo tamaño. DataMIL motiva evaluar utilidad para el desempeño, pero una heurística nuestra no debe llamarse implementación de DataMIL. [Dass et al.](https://arxiv.org/abs/2505.09603v2).

### 6.5 Frecuencia, redundancia y almacenamiento

Conservar inicialmente logs a la frecuencia física para auditar alineación. Elegir puntos de entrenamiento con submuestreo temporal, conservando suficientes ejemplos de arranque, frenado y cambio de fase. Analizar autocorrelación dentro de cada episodio y distribución de cambios de acción. El número efectivo de observaciones temporales puede usarse como diagnóstico, no como sustituto del número de escenarios independientes.

Separar cuatro decisiones:

1. **Integración física:** actualmente 1 ms en la configuración local.
2. **Actualización de acciones:** inicialmente cada paso; reducirla exige un experimento nuevo con acciones mantenidas entre actualizaciones.
3. **Selección de ejemplos de entrenamiento:** puede ser dispersa sin reducir la frecuencia de control.
4. **Muestreo del historial:** ventanas definidas en segundos, posiblemente con pasos más espaciados.

Submuestrear una señal continua para análisis espectral requiere considerar aliasing; filtrar causalmente cuando corresponda y documentar el retraso. Si se cambia la frecuencia de control, recolectar/validar etiquetas coherentes con ese régimen. Un torque mantenido 10 ms puede comportarse de forma distinta a uno actualizado cada 1 ms.

Formato candidato: HDF5 por episodios o archivos NumPy comprimidos con manifiesto JSON/CSV. Elegir según acceso a ventanas y tamaño medido; guardar floats con precisión documentada y no duplicar físicamente todas las ventanas solapadas. Los nombres y formatos son una propuesta, todavía no una API implementada.

| Grupo | Campos mínimos |
|---|---|
| Identidad | `episode_id`, `scenario_id`, `parent_episode_id` si hay derivados, semilla, split, hashes de código/config/dataset |
| Condición | Objetivo, pose inicial, velocidades iniciales, física, perturbaciones y resultado de calificación |
| Tiempo | `step`, `t`, `dt`, instante de observación y de aplicación de acción |
| Estado | Base, orientación completa, q/qd completos con nombres; observaciones extra disponibles |
| Acciones | `expert_wheel_cmds`, `expert_torque_arm`, acción realmente enviada y saturaciones; origen de la acción |
| Diagnóstico | EE verdadero, errores, fase experta, referencias internas, contactos y límites; campos separados de entradas permitidas |
| Final | Éxito según protocolo, razón de terminación, número de pasos, integridad y causa de exclusión si la hay |

### 6.6 Splits y prevención de fuga de información

Asignar escenarios a splits **antes de generar ventanas, normalizar o hacer aumentos de datos**. Todas las versiones de un episodio, trayectorias duplicadas, recuperaciones derivadas y ventanas pertenecen al mismo split. Si un conjunto está definido por objetivo reservado, todas las repeticiones de ese objetivo permanecen allí; para objetivo+postura, registrar además qué factores comparten soporte con entrenamiento.

Proponer 256 condiciones de validación de desarrollo, estratificadas, y suites de test independientes definidas en la sección 10. Si se ajusta robustez con validación perturbada, usar semillas y condiciones distintas de las del test. El conjunto piloto queda marcado como desarrollo y no se recicla como evaluación confirmatoria.

Calcular normalización, selección de características, filtros ajustados con datos y parámetros de Fourier usando solo entrenamiento. Los hiperparámetros y checkpoints se eligen con validación. El test se usa tras congelar las decisiones; nuevas ideas nacidas al verlo pasan a exploratorias o necesitan un nuevo conjunto confirmatorio.

No augmentar objetivos cambiando solo `TARGET_EE_POS`: la acción del experto cambiaría. No rotar o trasladar pares estado–acción sin demostrar la simetría de controlador y dinámica. No construir ventanas centradas que incluyan futuro. Las transformaciones válidas se aplican por episodio y conservan la trazabilidad.

## 7. Entradas y sensores: qué existe y qué debemos validar

Guardar muchas señales para diagnóstico no implica alimentar todas al modelo. La información de entrenamiento debe tener un equivalente disponible al ejecutar la política.

| Grupo | Señal y procedencia | Hipótesis y control necesario |
|---|---|---|
| F0: baseline | `base_pos`, `base_bearing`, `q_mes[4:11]`, `qd_mes[4:11]`, objetivo | 21 variables; repetir con seno/coseno del bearing. Conservar la referencia literal del esquema |
| F1: geometría | Objetivo relativo a la base; opcionalmente rotado al marco de la base | Puede facilitar aprendizaje; comparar añadirlo y reemplazar coordenadas. El experto no es automáticamente invariante al origen |
| F2: velocidades de base | `GetBaseLinVelocity`, `GetBaseAngVelocity`; proyección planar o marco corporal | Distinguir estar quieto de estar deslizándose. Medir coste y ruido; validar transformaciones de marco |
| F3: ruedas | `qd_mes[:4]` y errores entre comando previo y velocidad medida | Información sobre respuesta/deslizamiento. Ángulos absolutos acumulados de ruedas podrían actuar como reloj espurio; estudiarlos aparte |
| F4: aceleración articular | Diferencia causal `(qd_t - qd_prev)/dt`, con filtro causal opcional; existe getter con inconsistencia pendiente | Puede revelar transitorios, pero amplifica ruido. Comparar derivada ideal, derivada de señal ruidosa y versión filtrada |
| F5: aceleración/IMU de base | Diferencias causales de velocidad lineal y angular; orientación y giroscopio simulados | La aceleración del mundo no es por sí sola una lectura de acelerómetro. Una IMU simulada exige marco, gravedad, sesgo y ruido explícitos |
| F6: acciones anteriores | Últimas velocidades de ruedas solicitadas y torques del brazo enviados por la propia política | Aporta historia de control. Al evaluar deben ser sus propias acciones, no las acciones del experto; medir el desajuste de entrenamiento |
| F7: torque aplicado previo | `GetMotorTorques` leído antes de emitir la acción nueva, con timestamp validado | Es salida de la capa de actuadores. Para el brazo puede ser casi redundante con la acción anterior; no llamarlo medición independiente de fuerza |
| F8: cinemática del EE | Posición y error del EE calculados con Pinocchio a partir de mediciones | Aporta conocimiento geométrico, no necesariamente información nueva. Contar coste de FK; usar medidas coherentes con ruido, no ground truth oculto |
| F9: estado inicial/reloj | Postura inicial registrada, tiempo transcurrido o paso | Son disponibles, pero pueden crear dependencia de duración. Comparar con memoria y probar retrasos/velocidades diferentes |
| F10: contactos/fuerzas | Consulta de contactos de PyBullet; sensores de fuerza/torque solo tras configurar articulaciones válidas | Puede ayudar con interacción física. El archivo incluye nombres de pies: verificar su pertinencia al Robotnik; no asumir sensores operativos |
| F11: percepción | Cámara RGB/depth o rayos virtuales añadidos con APIs del simulador | Solo justificar si introducimos incertidumbre de localización u obstáculos. Con estado y objetivo exactos puede añadir coste sin resolver una limitación real |
| O: información privilegiada | Fase real del experto, `q_des_arm`, `qd_des_arm`, acción experta actual | Solo diagnóstico/oráculo. Las acciones actuales son etiquetas, nunca entradas del regresor que las predice |

En robótica real las posiciones articulares corresponderían a encoders, las velocidades podrían derivarse o estimarse, y la pose de base necesitaría localización/odometría. Aquí empezamos con estado de simulación. Comparar verdad de simulación, señal degradada y estimador, sin presentar una señal perfecta como sensor realista.

Para cada grupo de entradas añadirlo **por separado** a un baseline congelado. Después combinar los grupos que aporten una mejora reproducible. Si se añade F8, derivarlo de la observación que recibe el modelo: usar el EE verdadero del simulador bajo ruido podría eliminar artificialmente la dificultad.

Las etiquetas expertas pueden calcularse con estado limpio durante un experimento de observación ruidosa, si se declara que se aprende a actuar correctamente a partir de observaciones imperfectas. Eso difiere de imitar las decisiones de un experto que también ve ruido. Evaluar ambas definiciones por separado cuando sea relevante; en ninguna se entrega la etiqueta al modelo como característica.

## 8. Un instante, ventanas temporales y Fourier

### 8.1 Protocolo temporal

Comparar estado instantáneo; concatenación de observaciones pasadas; TCN causal; GRU/LSTM con memoria. Definir el historial en tiempo físico: propuestas de **20, 100, 250 y 500 ms**, con 1 s como extensión para fases largas. Elegir resolución de muestreo con el piloto; “20 muestras” cubre duraciones distintas a 1 kHz y a 100 Hz.

Cada ventana termina en el instante de decisión. Para comienzos de episodio usar padding con primera observación y máscara, o un estado inicial definido; aplicar la misma regla en entrenamiento e inferencia. Las ventanas nunca cruzan resets. En GRU/LSTM, reiniciar memoria al empezar un episodio y definir burn-in para entrenamiento por fragmentos. Entrenar con estados ocultos siempre a cero en fragmentos intermedios y evaluar con memoria continua puede crear otro desajuste.

Comparar memoria finita y memoria recurrente bajo parámetros similares. Una GRU que procesa todo el episodio tiene acceso a más historia que una ventana de 100 ms: explicitarlo, o añadir un control recurrente reiniciado con el mismo historial. Si la política necesita la postura inicial, comparar también F9; más muestras recientes no necesariamente la recuperan.

Separar **historial de observaciones** de **horizonte de acciones**. Usar información pasada puede mantener una decisión reactiva en cada paso. Predecir un bloque de acciones futuras introduce otra decisión: cuántas ejecutar antes de observar de nuevo.

### 8.2 Experimentos de Fourier con significado concreto

La propuesta de Fourier se descompone en tres líneas independientes:

1. **Análisis exploratorio:** espectro/PSD de velocidades, aceleraciones y torques de demostraciones de entrenamiento, por fase. Identificar oscilaciones, escalas temporales y contenido asociado al ruido. Esto orienta ventanas y muestreo; no prueba por sí mismo que las características espectrales ayuden.
2. **Características espectrales del historial:** calcular FFT real o DCT sobre una ventana pasada, con longitud y frecuencia conocidas. Usar pocos coeficientes o energía por bandas junto al estado actual. Comparar con media, desviación y diferencias temporales simples sobre la misma ventana.
3. **Fourier features de coordenadas:** codificar posiciones normalizadas con senos/cosenos de frecuencias elegidas usando entrenamiento/validación. Esto cambia la representación espacial y debe tener un experimento distinto. [Tancik et al.](https://arxiv.org/abs/2006.10739).

Para la línea 2: retirar media cuando proceda, documentar ventana de Hann u otra, frecuencia de muestreo y bandas; calcular únicamente con datos disponibles hasta $t$. La resolución espectral es aproximadamente $1/T_w$ y el límite de Nyquist $f_s/2$: una ventana de 100 ms no puede resolver con precisión un fenómeno de 1 Hz. Magnitud sola pierde fase y orden temporal; comparar coeficientes reales/imaginarios o DCT con signo y conservar la observación actual. Los movimientos de alcance son transitorios y no tienen por qué ser periódicos.

Un experimento de compresión adicional puede predecir coeficientes DCT de un bloque de **acciones futuras** y reconstruirlo. Es aprendizaje de una salida futura, por lo que esas acciones pueden ser etiquetas durante entrenamiento, pero nunca entradas al ejecutar. Mantener el resultado final en las 11 acciones continuas requeridas y evaluar el error de reconstrucción y la estabilidad. FAST usa DCT para tokenizar secuencias en VLAs; tomamos esa inspiración como motivación, sin afirmar que implementamos FAST. [Pertsch et al.](https://arxiv.org/abs/2501.09747).

**Control decisivo:** igual ventana disponible, dimensiones/capacidad comparables y coste de preprocesamiento incluido. Si una FFT mejora respecto al estado instantáneo pero no respecto a una ventana cruda, la evidencia favorece la historia temporal y no necesariamente la transformación espectral.

## 9. Arquitecturas del regresor y entrenamiento

### 9.1 Familia de modelos y función de cada comparación

Todos los baselines aprendidos reciben observaciones permitidas y producen acciones continuas. Primero asegurar que las unidades, normalización y lazo de control funcionan; después aumentar capacidad.

| Modelo | Configuración inicial propuesta | Qué permite estudiar |
|---|---|---|
| E | Experto original en el mismo evaluador | Referencia de desempeño y coste; no es un techo matemático |
| B0 | Predictor de media de acciones | Sanity check offline. No hace falta ejecutarlo físicamente en cada comparación |
| B1 | Regresión Ridge multisalida | Cuánto explica un modelo lineal con regularización |
| B2 | Extra Trees / random forest multisalida | Baseline no neuronal para estados de baja dimensión; medir discontinuidad, coste y extrapolación |
| B3 | MLP con 2 capas ocultas de 128 unidades, ReLU, salida lineal de 11 | Baseline neuronal principal, con versiones de 64/128/256 unidades y 2/3 capas |
| B4 | MLP con ventana aplanada | Efecto de observar historia sin una arquitectura temporal especializada |
| B5 | TCN causal, bloques residuales y dilataciones | Historia finita con estructura temporal; registrar campo receptivo real |
| B6 | GRU, 1–2 capas, estado oculto 64/128; LSTM como alternativa | Memoria del episodio y transiciones del experto |
| B7 | Tronco compartido con cabeza de ruedas y cabeza de torques; alternativa: redes separadas | Interferencia entre dos subsistemas con escalas y dinámicas distintas |
| B8 | Regressor condicionado por fase predicha, con cabezas especializadas | Estructura jerárquica aprendida; reportar errores de fase y de regresión. La fase verdadera queda como oráculo diagnóstico |
| B9 | Transformer temporal pequeño o predictor de bloques inspirado en ACT | Dependencias más largas y acciones futuras; controlar historial, parámetros y horizonte ejecutado |
| B10 | GMM/mezcla de expertos o Diffusion Policy condicionada por estado | Multimodalidad residual, solo si persiste tras corregir observabilidad y datos |
| B11 | Compresión/destilación de una política competente, incluida variante inspirada en OneDP | Frontera precisión–coste; contar también el coste de entrenar el profesor |

La elección inicial de MLP 2×128 es un punto de referencia reproducible. Con 22 entradas y 11 salidas tiene 20 875 parámetros si todas las capas son densas con bias. El recuento final se obtendrá del modelo construido, porque cabezas y entradas adicionales lo cambian.

B5 se apoya en la comparación de modelos temporales de Bai et al.; B6, en el interés de BC recurrente mostrado por Mandlekar et al. Son candidatos, no una expectativa de superioridad garantizada. [Bai et al.](https://arxiv.org/abs/1803.01271), [Mandlekar et al.](https://proceedings.mlr.press/v164/mandlekar22a.html).

ACT y Diffusion Policy amplían el menú contemporáneo de BC. Para este controlador, su adopción necesita evidencia de una limitación concreta del regresor: memoria, multimodalidad o coordinación temporal. La ambigüedad por ocultar el estado del experto no se resuelve necesariamente con un generador más complejo. Comparar acciones deterministas y estocásticas con semillas de inferencia registradas. [Zhao et al.](https://arxiv.org/abs/2304.13705), [Chi et al.](https://arxiv.org/abs/2303.04137).

### 9.2 Normalización y pérdidas

Normalizar entradas con estadísticas de entrenamiento; manejar columnas constantes sin dividir por cero. Usar seno/coseno para bearing y estudiar una codificación periódica de articulaciones donde tenga sentido físico. No envolver indiscriminadamente articulaciones con topes mecánicos.

Las salidas mezclan rad/s y N·m. Usar una pérdida por grupos con escalas $s_j$ obtenidas de entrenamiento, con un mínimo para canales casi constantes:

$$L_{BC}=\frac{\lambda_w}{4}\sum_{j\in W}\left(\frac{\hat a_j-a^E_j}{s_j}\right)^2+\frac{\lambda_\tau}{7}\sum_{j\in A}\left(\frac{\hat a_j-a^E_j}{s_j}\right)^2.$$

Punto de partida: $\lambda_w=\lambda_\tau=1$. Comparar MSE normalizada con Huber; estudiar ponderación por fase solo como una intervención explícita. Reportar siempre MAE/RMSE en unidades físicas. Comparar normalización estadística con escalas físicas de referencia si la varianza desigual hace que pequeños canales dominen la pérdida.

La regularización de suavidad requiere muestras consecutivas reales, no pares de un batch barajado. Penalizar variación absoluta puede borrar cambios legítimos de fase; una alternativa es penalizar diferencia entre incrementos de acción predichos y expertos. Evaluar su efecto en latencia, error y recuperación, no asumir que un comando más suave sea mejor.

### 9.3 Restricciones de salida

Mantener el límite de torque de ±30 N·m del controlador como referencia inicial, sujeto a comprobar los límites reales del robot. Las velocidades de ruedas deben tener un límite documentado derivado de la interfaz/experto, no del mínimo y máximo observados en un subconjunto pequeño. Guardar acción cruda, acción limitada y frecuencia de saturación; medir cuánto trabajo realiza el limitador.

Comparar cuatro velocidades libres con dos salidas izquierda/derecha expandidas al orden original. El segundo diseño incorpora una estructura verificada del experto; no debe ocultarse como mera reducción de parámetros. Mantener siete torques del brazo en ambas variantes.

Cualquier filtro, limitador de variación o capa de corrección forma parte del sistema evaluado y debe contar en el coste. Si se incorpora compensación gravitatoria analítica más torque residual aprendido, etiquetarlo como **variante híbrida** y compararlo con el regresor de torque total; no atribuir al modelo lo calculado por el componente físico. Durante el test principal no habrá sustitución silenciosa por acciones expertas.

### 9.4 Optimización, selección y comparabilidad

Propuesta de búsqueda neuronal: AdamW; tasa de aprendizaje logarítmica entre `1e-5` y `3e-3`; weight decay entre `0` y `1e-3`; batch de 128/256/512 ejemplos o ventanas; early stopping; clipping de gradiente para modelos recurrentes; semillas explícitas. Estos valores son candidatos a validar, no hiperparámetros óptimos conocidos. Ridge y árboles tendrán búsquedas propias comparables en esfuerzo, no una única configuración desfavorable.

Usar búsqueda aleatoria o bayesiana con un presupuesto de ensayos declarado por familia. Distinguir dos preguntas: “mejor desempeño con igual presupuesto de búsqueda” y “desempeño con recursos computacionales equivalentes”. Registrar número de actualizaciones, ejemplos vistos, tiempo de entrenamiento y política de evaluación de checkpoints. Con datasets de tamaños distintos, comparar convergencia y un control a actualizaciones constantes; un número fijo de épocas no iguala cómputo.

Seleccionar checkpoints mediante un conjunto de rollouts de **validación** fijo y una regla registrada: primero éxito, después menor error final y, si son prácticamente equivalentes, menor coste. La pérdida offline es un diagnóstico adicional. Para evitar selección por ruido, usar igual número de evaluaciones por método y confirmar finalistas en un conjunto de validación reservado distinto; test permanece cerrado.

Para conclusiones principales, al menos **5 semillas de entrenamiento**, ampliables a 10 si las diferencias son pequeñas o la variabilidad grande. Son objetivos del diseño, no una garantía automática de potencia estadística. Publicar todas las semillas y ejecuciones, no solo la mejor.

## 10. Evaluación: precisión y generalización

### 10.1 Separar predicción offline y ejecución autónoma

La evaluación offline usa observaciones de trayectorias expertas reservadas para comparar acciones. La evaluación en lazo cerrado reinicia el simulador en cada escenario y deja actuar al modelo con sus propias observaciones, acciones previas y memoria. No reproducir estados expertos paso a paso y llamarlo control autónomo.

| Nivel | Métrica | Definición y propósito |
|---|---|---|
| Offline | MAE y RMSE por salida | Error respecto a cada velocidad/torque experto en sus unidades; macro promedio por articulación y por grupo |
| Offline | Error normalizado | Facilita comparar escalas; normalizadores fijados con entrenamiento |
| Offline | Error por fase y transición | Detecta si la media oculta fallos al girar, frenar o empezar a alcanzar |
| Lazo cerrado, primaria | Tasa de éxito `SR_1cm` | Fracción de episodios con distancia EE–objetivo ≤1 cm durante una permanencia definida, dentro del horizonte |
| Lazo cerrado | `SR_2cm`, `SR_5cm` y curva éxito–tolerancia | Sensibilidad al umbral; 2 y 5 cm son diagnósticos propuestos, no sustituyen silenciosamente el umbral principal |
| Lazo cerrado | Distancia final y mínima | Error terminal y mejor aproximación; reportar ambos para distinguir llegada fugaz de estabilidad |
| Lazo cerrado | Error integrado | $\frac{1}{T}\int_0^T\|p^{EE}_t-g\|_2dt$, con horizonte común para comparación |
| Lazo cerrado | Tiempo hasta éxito | Tiempo físico hasta primer periodo de llegada estable; los fallos quedan censurados o penalizados explícitamente |
| Lazo cerrado | Estabilidad de llegada | Deriva y variación del EE durante una ventana de mantenimiento |
| Lazo cerrado | Uso de actuadores | Frecuencia de saturación, picos, variación total normalizada y violaciones de límites |
| Lazo cerrado | Fallos de ejecución | NaN, divergencia, timeout, contactos definidos o condición terminal inválida |

**Definición propuesta de éxito:** distancia ≤1 cm durante 0.5 s antes de un horizonte de 40 s, sin terminación inválida. El centímetro proviene de la tolerancia configurada del experto; la permanencia es una decisión nuestra y debe calibrarse y congelarse. Reportar además una métrica de llegada compatible con la lógica original para distinguir el efecto del nuevo criterio. Un centímetro es el criterio del código inspeccionado, no un requisito numérico explícito de las diapositivas.

El evaluador externo no depende de `base_state.done`: el modelo aprendido puede no tener ese estado. Una política que alcanza antes de 8 s debe medirse correctamente; la restricción de 8 s pertenece a la parada del script original y se registra aparte. Evitar que la terminación por estancamiento dé a una política menos oportunidades que a otra. Para distancias terminales y error integrado usar horizonte común; si hay terminación inválida, conservar el fallo y aplicar una regla de penalización previamente definida, sin rellenar con valores favorables.

Medir el EE con estado verdadero para evaluar, independientemente del ruido que ve la política. Para estabilidad tras alcanzar el objetivo, continuar la simulación durante la ventana definida y comprobar que el controlador no solo cruza la zona de éxito por inercia.

No promediar directamente velocidad de ruedas con torque. No usar $R^2$ como métrica principal en canales casi constantes. La integral $\int|\tau^\top\dot q|dt$ puede informar sobre trabajo mecánico del brazo, pero no representa consumo eléctrico total ni eficiencia energética de toda la plataforma. Parámetros y memoria son métricas distintas del consumo de energía.

### 10.2 Suites separadas de generalización

**ID** significa condiciones nuevas muestreadas de la misma distribución de entrenamiento. **OOD** significa factores fuera del soporte de entrenamiento declarado. No llamar “extrapolación” a una repetición con otra semilla.

| Suite | Diseño | Interpretación |
|---|---|---|
| G0: ID | Objetivos y condiciones nuevos dentro del mismo dominio, sin duplicados | Interpolación y generalización dentro de distribución |
| G1: regiones reservadas | Sectores/volúmenes completos de objetivos excluidos de entrenamiento; frontera de separación explícita | Generalización espacial a huecos no vistos; no necesariamente fuera de la envolvente global |
| G2: objetivos fuera del rango | Radios o alturas fuera de los rangos de entrenamiento, calificados antes por el experto | Extrapolación. Propuestas iniciales: radio 1.4–1.8 m o altura 0.8–1.05 m, sujetas al piloto |
| G3: condiciones iniciales nuevas | Objetivos ID con poses de base, yaw y posturas de brazo fuera de rangos de entrenamiento | Generalización de estado inicial, aislada del objetivo |
| G4: combinaciones nuevas | Factores individualmente vistos, pero ciertas combinaciones objetivo–postura reservadas | Generalización composicional |
| G5: cambio combinado | Objetivos y estados iniciales OOD simultáneos | Estrés conjunto; no permite atribuir el fallo a un solo factor |

Publicar el dominio de cada suite, manifest de escenarios, distribución de distancias/alturas y resultados por sector. En G3 variar un factor cada vez antes de combinarlos. Los objetivos inalcanzables o difíciles para el experto pueden formar una suite de estrés aparte; no contaminan una afirmación de imitación exitosa en un dominio calificado.

Evaluar el experto en exactamente los mismos escenarios y reportar `SR_modelo`, `SR_experto` y diferencia absoluta. Reportar resultados sobre el conjunto completo y sobre el subconjunto calificado definido de antemano; nunca filtrar por éxito del modelo. Si el experto falla fuera de distribución, discutir su límite y comprobar si los fallos del modelo coinciden.

### 10.3 Tamaño de evaluación e inferencia estadística

Para una política fija y escenarios Bernoulli independientes, la aproximación conservadora al tamaño necesario para estimar una tasa con semiancho $\epsilon$ al 95% es:

$$n\approx\frac{1.96^2\,0.25}{\epsilon^2}.$$

Esto da aproximadamente 97 escenarios para ±10 puntos porcentuales, 196 para ±7 y 385 para ±5. **Es precisión de una proporción, no potencia para detectar diferencias entre modelos ni para generalizar sobre semillas de entrenamiento.** Usar intervalos de Wilson para la tasa de una política; las aproximaciones de tamaño se revisan con el piloto y la dependencia del muestreo.

Propuesta confirmatoria: **400 escenarios por suite principal**, evaluados por las mismas 5 semillas de cada finalista; conjuntos más pequeños para cribado de validación. No tratar 2 000 rollouts como 2 000 observaciones totalmente independientes si comparten 400 escenarios y 5 políticas entrenadas.

Usar escenarios y perturbaciones emparejados entre modelos; reportar diferencias de éxito y error con intervalos del 95%. Para la comparación de métodos, remuestrear semillas de entrenamiento y escenarios en sus respectivos niveles, manteniendo el emparejamiento y agrupando escenarios derivados por familia. Si todas las semillas se prueban en todos los escenarios, el diseño es cruzado: no aplicar un bootstrap de filas que ignore esa estructura. Para una sola pareja de políticas fijas, McNemar puede complementar la comparación binaria emparejada.

Para error final: mediana, rango intercuartílico y media con intervalo; mostrar distribución y colas. Para éxito: porcentaje con numerador/denominador. Para tiempo: éxito acompañado de tiempo entre éxitos, más curva de probabilidad de haber alcanzado el objetivo antes de $t$ o tiempo penalizado por horizonte. Una política que solo resuelve casos fáciles no debe parecer más rápida por excluir sus fallos.

Predefinir comparaciones primarias, por ejemplo B3 frente al mejor modelo temporal y F0 frente a F0+velocidades. En familias de contrastes confirmatorios aplicar ajuste como Holm y reportar tamaños de efecto. Las numerosas ablaciones exploratorias se presentan como tales. Un valor p no sustituye relevancia práctica ni intervalos de incertidumbre. El énfasis en incertidumbre está motivado por [Agarwal et al.](https://arxiv.org/abs/2108.13264); el diseño estadístico concreto de este proyecto es nuestra adaptación.

## 11. Robustez y aprendizaje de recuperación

### 11.1 Perturbaciones controladas

Aplicar una dificultad cada vez, después combinaciones justificadas. Congelar modelos antes del test. Usar exactamente la misma perturbación de escenario para comparar experto y modelos; distinguir ruido de observación, perturbación de acción y cambio de física.

| Factor | Niveles iniciales para calibrar | Qué se registra |
|---|---|---|
| Ruido de posición | Desviación 0 / 1 / 5 / 10 mm | Ruido instantáneo frente a sesgo fijo por episodio; separar base, objetivo y EE derivado |
| Ruido angular | Desviación 0 / 0.1 / 0.5 / 1 grado, convertido a radianes | Aplicar orientación de forma válida, no perturbar cuaterniones sin normalización |
| Ruido de velocidad | Escalas físicas determinadas con piloto y, como análisis secundario, 1% / 5% / 10% de escala de entrenamiento | Efecto en velocidades y derivadas; declarar la definición de escala |
| Retardo de observación | 0 / 1 / 5 / 10 / 20 ms | Buffer causal, arranque del buffer y timestamps |
| Pérdida de observaciones | Probabilidad 0 / 1% / 5% / 10%, además de ráfagas | Mantener última observación, máscara y antigüedad; misma estrategia para todos |
| Fricción | Multiplicadores 0.5 / 0.75 / 1 / 1.25 del nominal | Fricción de suelo y ruedas por separado; coeficiente efectivo y deslizamiento |
| Empujones | Impulsos con dirección, momento y magnitud fijos por escenario | Calibrar impulso como masa × cambio de velocidad deseado; tiempo de recuperación y tasa de éxito |
| Masa/carga | Multiplicadores 0.9 / 1 / 1.1 / 1.2 donde sean físicamente coherentes | Actualizar inercias o añadir una carga modelada; declarar si el experto conoce el cambio |
| Respuesta de actuadores | Ganancia o retraso de comando | Saturación, oscilación y divergencia; no confundir con ruido del sensor |
| Frecuencia de control | 1 000 / 500 / 200 / 100 Hz como candidatos | Integración física mantenida; acciones sostenidas y compatibilidad del experto/modelo |

Los rangos anteriores son para el piloto, no tolerancias garantizadas del hardware. Registrar unidades y mecanismo de cada perturbación. La configuración nativa de ruido puede afectar también al servo interno de ruedas porque este consulta getters; para un experimento de ruido solo en la política, usar una capa de observación explícita que no cambie la física ni el servo.

Métricas: éxito y error frente a intensidad, caída absoluta respecto al nominal, peor subgrupo y área bajo la curva de robustez con normalización declarada. No sumar en un único número curvas con unidades distintas sin definir sus pesos. Probar perturbación durante navegación y durante alcance por separado; empujar después de `done` puede exponer que el experto no reactiva la base.

### 11.2 Robustez adquirida frente a robustez incidental

Comparar entrenamiento nominal con entrenamiento que contiene ruido/variación de física y con demostraciones de recuperación. Los niveles vistos en entrenamiento y los niveles reservados deben declararse. Evaluar también el rendimiento nominal para detectar si se gana robustez a costa de precisión.

En recolección con perturbación de acciones: consultar la acción correctiva limpia del experto en el estado actual, guardar esa etiqueta y aplicar una acción perturbada controlada para explorar; en el siguiente estado el experto vuelve a decidir. Distinguir etiqueta experta de acción ejecutada. Si no se optimiza la distribución de perturbación como en DART, llamar al método **BC con demostraciones perturbadas**, inspirado en DART. [Laskey et al.](https://proceedings.mlr.press/v78/laskey17a.html).

### 11.3 DAgger como extensión supervisada interactiva

Entrenar BC inicial; ejecutar la política en escenarios de entrenamiento; consultar al experto en estados visitados; agregar pares observación–corrección; volver a entrenar. Contabilizar episodios y consultas adicionales y comparar con BC que recibe el mismo presupuesto de nuevas demostraciones. No usar escenarios de test para recolección.

El experto tiene estado mutable: definir una instancia por episodio, inicializada con la condición real y actualizada una sola vez en cada paso observado aunque no se ejecute su acción. Guardar fase, tiempo y referencia interna con cada consulta. Una segunda llamada para logging puede mutar la fase o integrar dos veces la referencia del brazo, alterando las etiquetas. Verificar que sus correcciones son válidas en estados alejados y registrar cuándo no sabe recuperar. DAgger no convierte automáticamente a este experto en un controlador óptimo de recuperación. [Ross et al.](https://proceedings.mlr.press/v15/ross11a.html).

## 12. Eficiencia y frontera de desempeño

Medir **tiempo de decisión**, **tiempo físico hasta llegar** y **tiempo de entrenamiento** por separado. Una inferencia rápida puede generar movimientos lentos o poco precisos.

| Medida | Protocolo |
|---|---|
| Parámetros | Recuento entrenable y total; en árboles añadir nodos/profundidad/tamaño, sin equipararlos a pesos neuronales |
| Tamaño de pesos | Bytes del checkpoint y precisión numérica; un FP32 denso requiere aproximadamente 4 bytes por parámetro solo para pesos |
| Memoria en ejecución | Pico y memoria estable del proceso/modelo, buffers temporales y activaciones; separar CPU y GPU |
| Latencia del modelo | Batch 1, calentamiento, miles de llamadas, mediana/p95/p99; CPU primero y GPU por separado |
| Latencia completa | Lectura/procesamiento, FK si se usa, ventanas/FFT, transferencias, inferencia, desnormalización y límites |
| Plazo de control | Fracción de decisiones que exceden `dt_control`; capacidad de atender 1 kHz no se infiere de una media inferior a 1 ms |
| Coste de aprendizaje | Recolección, entrenamiento, búsqueda y consultas expertas; hardware, threads, precisión y versiones |

Excluir renderizado, plots y escritura de logs del benchmark de inferencia, pero medir aparte el coste total de simulación. Cronometrar experto y modelo con los mismos límites de medición, mismas condiciones y política de threads; sincronizar GPU al medir y distinguir throughput por lotes de latencia por decisión. Si se usa el EE solo para evaluación, separar ese cálculo común del coste de la política; si es una entrada, incluirlo.

Construir frentes de Pareto: éxito frente a p95/p99, memoria y parámetros, con incertidumbre en éxito. Definir una pérdida práctica tolerable, por ejemplo 2 puntos de éxito, antes de declarar un modelo “equivalente pero más eficiente”; comprobarlo mediante intervalos o prueba de no inferioridad preespecificada, no por ausencia de significación.

Explorar reducción de anchura, destilación, precisión reducida o cuantización después de establecer un modelo competente. Cada versión comprimida requiere nuevas ejecuciones en lazo cerrado: pequeños errores numéricos pueden afectar estabilidad. OneDP muestra una dirección reciente para acelerar políticas generativas; sus frecuencias publicadas no se transfieren a nuestra tarea. [Wang et al.](https://proceedings.mlr.press/v267/wang25ba.html).

## 13. Matriz de experimentos y controles

El orden evita atribuir a una arquitectura lo que en realidad viene de corregir el dataset o de añadir información. Las extensiones se mantienen en el plan amplio y se activan cuando su pregunta esté bien planteada.

| ID | Experimento | Control que se mantiene | Resultado principal | Dependencia |
|---|---|---|---|---|
| E00 | Calificar experto, señales y evaluador | Starter nominal y versiones congeladas | Mapa de éxito del experto y auditoría temporal | Inicio |
| E01 | Media, Ridge, árboles y MLP con F0 | Mismos episodios, splits e interfaz | Baseline offline y éxito autónomo | E00 |
| E02 | Curvas de 32 a 4 096 episodios | Validación fija; MLP y temporal de referencia | Ganancia por duplicar datos, con incertidumbre | E01 |
| E03 | Diversidad frente a repetición | Episodios equiparados; control extra por transiciones/cómputo | Generalización G0–G4 | E01 |
| E04 | Calidad y recuperación Q0–Q3 | Tamaño/cobertura y presupuesto registrados | Éxito nominal y robustez | E01 |
| E05 | Entradas F1–F9, añadidas individualmente | Datos y modelo base congelados | Beneficio de cada grupo, ideal y con ruido | E01 |
| E06 | Instante, ventanas, TCN, GRU | Entradas, duración histórica y capacidad comparables cuando proceda | Ganancia atribuible a memoria | E01, auditoría de fases |
| E07 | Ventana cruda, resúmenes simples, FFT/DCT | Mismo historial y coste incluido | Información temporal frente a transformación espectral | E06 |
| E08 | Cabezas, redes separadas y simetría de ruedas | Datos/entradas y cómputo comparables | Coordinación y coste de la estructura | E01 |
| E09 | Tamaño y arquitectura finalista | Presupuesto de búsqueda y reglas de selección comunes | Frontera éxito–latencia–memoria | E02, E05, E06 |
| E10 | Robustez nominal frente a entrenamiento perturbado | Condiciones y perturbaciones de test emparejadas | Curvas por severidad y coste nominal | E04, E09 |
| E11 | BC + recolección aleatoria frente a DAgger | Igual presupuesto de consultas/datos adicionales | Recuperación y desplazamiento de distribución | E09, experto validado fuera de trayectoria |
| E12 | Fourier features espaciales o representación espectral de acciones | Baseline de representación y horizonte de acciones fijo | Aporte específico de representación | E07 o E09, según variante |
| E13 | ACT inspirado, GMM/difusión y destilación | Misma interfaz, información, test y presupuesto declarado | ¿La complejidad adicional resuelve un fallo observado? | E09, diagnóstico de memoria/multimodalidad |
| E14 | Evaluación confirmatoria de finalistas | Decisiones congeladas, todas las semillas y suites | Resultados para el informe | E09–E13 según alcance final |

Primero realizar ablaciones de un factor. Después estudiar interacciones concretas mediante un diseño factorial pequeño: **memoria × velocidades extra**, **aceleración × ruido** y **cobertura × capacidad**. La mejora de dos intervenciones juntas no demuestra que ambas sean necesarias; incluir cada una individualmente y una ablación de eliminación desde la combinación final.

Los modelos que reciben fase experta o referencias internas se rotulan como oráculos diagnósticos. Un controlador híbrido, un modelo con FK y uno que solo usa estado bruto pertenecen a configuraciones informativas distintas. Mostrar esas diferencias junto a sus resultados y costes.

## 14. Ingeniería experimental y reproducibilidad

### 14.1 Estructura prevista

Los siguientes directorios y comandos describen artefactos por implementar, no herramientas ya disponibles:

```text
docs/PLAN_DE_INVESTIGACION.md       # este protocolo
experiment_specs/                 # escenarios, barridos, entradas y reglas de evaluación
src/env/                          # adaptación del simulador y reset reproducible
src/data/                         # registro, validación, splits y ventanas
src/features/                     # transformaciones causales
src/policies/                     # experto adaptado y regresores
src/train/                        # entrenamiento y selección
src/eval/                         # métricas, suites, robustez y benchmarks
data/                             # datos grandes, excluidos de Git
checkpoints/                      # pesos, excluidos de Git
results/                          # métricas compactas, manifiestos y figuras
report/                           # LaTeX IEEE, bibliografía y contribuciones
```

Mantener el starter identificable y describir modificaciones. Para el entorno, usar el raíz de RoboEnv al resolver `configs/` y `models/`, conforme a sus instrucciones locales, en vez de crear copias divergentes de assets. Registrar versiones y hashes también de URDF/configs, modificaciones locales, bibliotecas y hardware. Una actualización de `upstream` produce una versión experimental nueva si cambia experto, física o tarea; no mezclar datasets sin identificarlos.

### 14.2 Contrato de un paso de simulación

```text
1. Leer una observación coherente y marcar t.
2. Construir entradas exclusivamente con información disponible hasta t.
3. Consultar UNA vez al experto si corresponde recolectar una etiqueta.
4. Elegir la acción ejecutada: experto, estudiante o perturbación definida.
5. Guardar observación, etiqueta, acción cruda y acción aplicada con sus orígenes.
6. Avanzar la física según el periodo de control.
7. Leer el nuevo estado y actualizar métricas externas y terminales.
8. Guardar transición y estado final aunque el episodio termine.
```

En test del estudiante, el experto no genera acciones de respaldo. Puede ejecutarse una trayectoria experta independiente desde el mismo reset para la comparación. El evaluador dispone de verdad de simulación; la política solo de sus entradas declaradas.

### 14.3 Comprobaciones que validan la metodología

- Verificar correspondencia de nombres/índices, signos y unidades con un episodio inspeccionado; confirmar que la etiqueta precede a la transición que provoca.
- Reproducir una trayectoria guardada mediante acciones registradas y comparar su evolución dentro de tolerancias definidas; documentar si la simulación no es bit a bit determinista.
- Comprobar resets, semillas, ausencia de estados recurrentes o referencias expertas heredadas entre episodios y cierre del cliente de simulación.
- Detectar duplicados y parentesco entre episodios; comprobar que no hay cruces de splits ni ventanas que atraviesen resets.
- Comparar aceleración calculada con diferencias conocidas y timestamps; validar rama de ruido/retraso antes de recolectar masivamente.
- Verificar que normalizadores y extracción temporal dan el mismo resultado durante entrenamiento y ejecución usando un prefijo idéntico.
- Probar invariantes de causalidad: modificar datos posteriores a t no debe cambiar la entrada ni la predicción en t.
- Verificar métricas con trayectorias sencillas: entrada fugaz a la zona de éxito, permanencia, timeout y fallo numérico.
- Confirmar que las acciones anteriores usadas por el estudiante son las propias, y que no hay acceso accidental a fase, referencia o acción experta actual.
- Reproducir un resultado desde un checkout limpio usando configuración, dataset y checkpoint identificados.

Las comprobaciones se diseñarán sobre estos riesgos reales; no se requiere construir una infraestructura extensa antes de comprobar el primer episodio.

### 14.4 Registro de cada ejecución

Guardar `run_id`, commit y diff local, hash de especificación, dataset/split, semilla de datos/entrenamiento/evaluación, entradas, arquitectura, hiperparámetros, presupuesto de búsqueda, checkpoint seleccionado, hardware y métricas por episodio. Derivar tablas y figuras de esos archivos mediante scripts; evitar copiar cifras manualmente.

Un registro de decisión debe tener: pregunta, alternativa considerada, evidencia de validación, decisión y limitación. Registrar resultados negativos, ejecuciones fallidas y cambios de protocolo. Las contribuciones del equipo se documentarán durante el trabajo con tareas, experimentos y revisión; el número de commits por sí solo no mide contribución científica.

## 15. Fases de ejecución y criterios para avanzar

| Fase | Trabajo | Evidencia de salida |
|---|---|---|
| P0: contrato y experto | Entender controlador, interfaz, assets y criterio de éxito | Episodio nominal auditado y lista de supuestos; dominio piloto descrito |
| P1: banco experimental | Reset, escenarios, registro y evaluador externo | Reproducción de trayectorias y pruebas de causalidad/alineación superadas |
| P2: datos piloto y BC | Dataset pequeño, splits, Ridge/MLP | Curvas de aprendizaje iniciales y primeras ejecuciones autónomas; fallos clasificados |
| P3: datos e información | Cantidad/calidad/cobertura y grupos de entradas | Evidencia de qué mejora desempeño y por qué; datos versionados |
| P4: memoria y representación | Ventanas, recurrencia, TCN y Fourier | Comparaciones que distinguen historia, capacidad y representación |
| P5: robustez y coste | Perturbaciones, recuperación y Pareto | Curvas de degradación, memoria, parámetros y latencias completas |
| P6: extensiones fundamentadas | DAgger, jerarquía, generativos o compresión según el fallo observado | Valor adicional frente a baselines con presupuesto y observaciones declarados |
| P7: confirmación y comunicación | Congelar finalistas, ejecutar suites y redactar | Tablas con intervalos, figuras reproducibles, demo, informe y contribuciones |

Si P2 falla, diagnosticar primero: sincronización, normalización, orden de articulaciones, cobertura, estado interno no observado y despliegue de acciones. Una pérdida offline baja junto a fracaso autónomo exige mirar trayectorias y distribución visitada. Aumentar tamaño de red no es una explicación del fallo.

Este plan conserva el programa completo solicitado. La selección posterior de experimentos para entrega debe preservar un baseline, controles causales, validación separada y cobertura de los cuatro ejes de evaluación, aunque se reduzcan algunas extensiones.

## 16. Figuras, tablas y relato científico del informe

El informe final debe sostener una afirmación central con evidencia. Formulación candidata, pendiente de resultados: **el desempeño de BC para este manipulador está condicionado por la cobertura de demostraciones y por la información temporal disponible, y esas decisiones producen compromisos medibles entre generalización, robustez y coste**.

Artefactos deseables:

1. Diagrama de experto → dataset → regresor → simulación, con entradas y salidas exactas.
2. Mapa del dominio de entrenamiento y suites reservadas, junto al mapa de desempeño del experto.
3. Curva de aprendizaje con bandas de incertidumbre y comparación diversidad/repetición.
4. Tabla de ablación de entradas y memoria, incluyendo baseline y tamaños de modelo.
5. Éxito/error por suite de generalización y curvas de robustez.
6. Frontera éxito–latencia–memoria y análisis de unos fallos representativos seleccionados con reglas explícitas.

Distribución tentativa de las 6 páginas: 0.75 para problema y contexto; 2 para metodología; 2.5 para resultados y discusión; 0.75 para limitaciones, conclusión y bibliografía. Ajustar al formato real, reservando espacio para referencias dentro del límite salvo aclaración del profesor. La declaración de contribuciones va aparte, máximo 1 página.

La discusión debe explicar **por qué** una variante funciona o falla: error en cambios de fase, falta de recuperación, cobertura angular, dependencia de postura inicial, sensibilidad a derivadas o coste de representación. Incluir diferencias relevantes aunque no favorezcan la hipótesis inicial. Un vídeo ilustra; las métricas por escenarios sustentan la conclusión.

## 17. Incertidumbres y decisiones pendientes

| Cuestión | Cómo resolverla |
|---|---|
| ¿Qué entiende exactamente la asignatura por `q_mes`/`qd_mes` en la entrada mínima? | Usar brazo según el esquema de las diapos; documentar que el array del script tiene 17 elementos y hacer una variante con velocidades de ruedas. Confirmar en briefing si hace falta |
| ¿Qué rango de objetivos y condiciones iniciales es razonable? | Piloto y calificación del experto, con registro de intentos fallidos; no inventar alcance garantizado |
| ¿Es válida una acción del experto en estados visitados por el modelo? | Auditoría de memoria y recuperaciones antes de DAgger; comparar casos con fase irreversible |
| ¿Aceleración aporta información o principalmente ruido? | Ablación causal con derivadas validadas y barrido de ruido |
| ¿Hay sensor de torque independiente? | La API inspeccionada entrega comandos aplicados; cualquier medición física simulada adicional exige otra implementación y validación |
| ¿Una historia breve basta para recuperar fases? | Comparación de ventanas, recurrencia y postura inicial; oráculo solo como diagnóstico |
| ¿Qué frecuencia de control es viable? | Medir experto/modelo y estabilidad antes de decimar acciones |
| ¿Cómo influyen colisiones y marcos de coordenadas? | Inspeccionar URDF y configuración efectiva; validar FK frente al simulador y autocollision si se utiliza como métrica |
| ¿Se admiten variantes híbridas, DAgger o generativas en el entregable principal? | Las diapos fijan BC supervisado y un regresor; conservar ese núcleo y presentar extensiones claramente. Llevar la cuestión al briefing si se convierten en el foco |
| ¿Cuándo puede afirmarse una ventaja sobre el experto? | Solo en el dominio, métrica y presupuesto evaluados, con incertidumbre; no extrapolar a superioridad general o robot real |

## 18. Bibliografía anotada y fuentes

Consulta web realizada el **16/10/2026**. Se priorizan artículos originales, actas y páginas de los autores. Las versiones arXiv se identifican cuando son relevantes; una prepublicación reciente no equivale a validación independiente. Las anotaciones indican para qué se usa cada referencia, no son resultados del proyecto.

1. **Vasconcelos, F.; Modugno, V.** *COMP0245 Assignment — 2026/27*. [Diapositivas locales](COMP0245-Assignment-2026-2.pdf). Fuente de alcance, preguntas, evaluación y entregables. SHA-256: `c05896d589255704ecee21b1f7932a6483628aa507c2ae1a68c1ccf52aa0a8f4`.
2. **Modugno, V.** [Starter público COMP0245](https://github.com/VModugno/LAB_SESSION_COMP_0245_2026_PUBLIC/tree/cae4fb69386872ae73c601ce77e182aad7abe284/final/robotnik_arm_task). Código de referencia inspeccionado. [RoboEnv](https://github.com/VModugno/RoboEnv) y [simulation_and_control](https://github.com/VModugno/simulation_and_control) proporcionan el entorno; versiones locales en la sección 3.
3. **Ross, S.; Gordon, G.; Bagnell, D. (2011).** *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*. AISTATS, PMLR 15:627–635. [Artículo y PDF](https://proceedings.mlr.press/v15/ross11a.html). Fundamento de DAgger y de la diferencia entre pérdida supervisada y comportamiento secuencial.
4. **Mandlekar, A. et al. (CoRL 2021; actas 2022).** *What Matters in Learning from Offline Human Demonstrations for Robot Manipulation*. PMLR 164:1678–1690. [Artículo y PDF](https://proceedings.mlr.press/v164/mandlekar22a.html). Consultadas metodología y comparaciones de observaciones, memoria, cantidad/calidad y selección de modelos.
5. **Lin, F.; Hu, Y.; Sheng, P.; Wen, C.; You, J.; Gao, Y. (2024, revisión 2026).** *Data Scaling Laws in Imitation Learning for Robotic Manipulation*. arXiv:2410.18647, v4, 26/06/2026. [Versión consultada](https://arxiv.org/abs/2410.18647v4). Fundamenta el experimento de diversidad frente a repetición y la necesidad de medir rendimientos marginales.
6. **Dass, S.; Khaddaj, A.; Engstrom, L.; Madry, A.; Ilyas, A.; Martín-Martín, R. (2025, revisión 2026).** *DataMIL: Selecting Data for Robot Imitation Learning with Datamodels*. arXiv:2505.09603, v2, 19/06/2026. [Versión consultada](https://arxiv.org/abs/2505.09603v2). Referencia reciente para selección de datos orientada al desempeño.
7. **Laskey, M.; Lee, J.; Fox, R.; Dragan, A.; Goldberg, K. (2017).** *DART: Noise Injection for Robust Imitation Learning*. CoRL, PMLR 78:143–156. [Artículo y PDF](https://proceedings.mlr.press/v78/laskey17a.html). Motiva recolectar correcciones después de perturbaciones.
8. **Bai, S.; Kolter, J. Z.; Koltun, V. (2018).** *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*. arXiv:1803.01271. [Artículo](https://arxiv.org/abs/1803.01271). Referencia de TCN frente a modelos recurrentes.
9. **Zhao, T. Z.; Kumar, V.; Levine, S.; Finn, C. (2023).** *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*. RSS; arXiv:2304.13705. [Artículo](https://arxiv.org/abs/2304.13705). Referencia de ACT para estudiar bloques de acciones, manteniendo las diferencias de interfaz con este proyecto.
10. **Chi, C. et al. (2023; versión extendida 2024).** *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*. Trabajo original RSS 2023; arXiv:2303.04137v5 es la versión extendida consultada. [Artículo](https://arxiv.org/abs/2303.04137v5). Referencia de BC generativo y horizonte recedente.
11. **Wang, Z. et al. (2025).** *One-Step Diffusion Policy: Fast Visuomotor Policies via Diffusion Distillation*. ICML, PMLR 267:63399–63416. [Artículo y PDF](https://proceedings.mlr.press/v267/wang25ba.html). Referencia de aceleración mediante destilación.
12. **Pertsch, K. et al. (2025).** *FAST: Efficient Action Tokenization for Vision-Language-Action Models*. arXiv:2501.09747. [Artículo](https://arxiv.org/abs/2501.09747). Inspiración para representación espectral de secuencias de acciones; no justifica por sí sola FFT de observaciones.
13. **Tancik, M. et al. (2020).** *Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains*. NeurIPS 33. [Artículo](https://arxiv.org/abs/2006.10739). Base conceptual de la ablación de codificación espacial.
14. **Agarwal, R.; Schwarzer, M.; Castro, P. S.; Courville, A.; Bellemare, M. G. (2021).** *Deep Reinforcement Learning at the Edge of the Statistical Precipice*. NeurIPS. [Artículo](https://arxiv.org/abs/2108.13264). Referencia de incertidumbre y evaluación con múltiples ejecuciones; adaptación explícita al diseño de BC.
15. **Shi, X.; Ebrahimi, A.; Hu, Y.; Jin, J. (2026).** *FODMP: Fast One-Step Diffusion of Movement Primitives Generation for Time-Dependent Robot Actions*. Preprint arXiv:2603.24806, 25/03/2026. [Artículo](https://arxiv.org/abs/2603.24806). Contexto reciente sobre velocidad y estructura temporal; lectura exploratoria del resumen, no base de una decisión de implementación.

## 19. Glosario para discutir el proyecto

| Término | Significado aquí |
|---|---|
| Experto | Programa del profesor que proporciona acciones de referencia |
| BC / behaviour cloning | Ajustar un modelo supervisado a ejemplos de lo que observó e hizo el experto |
| Regresor | Modelo que predice números continuos: velocidades y torques |
| Política | Regla que transforma observaciones, y posiblemente memoria, en acciones |
| Trayectoria / episodio | Ejecución completa desde un reset hasta su terminación |
| Rollout | Ejecutar una política en el simulador y observar su evolución |
| Lazo cerrado | Volver a observar el estado que producen las propias acciones antes de decidir |
| Ablación | Cambiar o quitar un componente para medir su aportación |
| Generalización | Desempeño en condiciones que no se usaron para entrenar |
| Covariate shift | Cambia la distribución de estados observados; en BC puede ocurrir porque el modelo se desvía del experto |
| Fuga de información | Usar inadvertidamente futuro, etiquetas o información del test para aprender o decidir |
| Información privilegiada | Señal disponible para el simulador o el experto que no se admite como entrada del estudiante |
| Parámetros | Valores aprendidos por un modelo; hiperparámetros son decisiones sobre arquitectura y entrenamiento |
| Pareto | Opciones para las que no se puede mejorar un criterio sin empeorar al menos otro |
| Resultado confirmatorio | Prueba de una comparación decidida antes de consultar el conjunto de evaluación final |
