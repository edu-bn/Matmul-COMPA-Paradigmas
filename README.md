# Matmul-COMPA

## Hardware utilizado

Para la ejecución de las pruebas de rendimiento se utilizó una instancia en la nube de Google Colab con las siguientes características:

    Procesador (CPU): Intel(R) Xeon(R) CPU @ 2.00GHz
    Arquitectura: x86_64
    Tarjeta Gráfica (GPU): NVIDIA Tesla T4  
    Memoria de Video (VRAM): 15360 MiB (Aprox. 15 GB)
    Arquitectura: Turing
    Capacidad de Cómputo (Compute Capability): 7.5

## Resultado y analisis
    N=2048 ALG=1 Threads=4
      Tiempo: 87707.5 ms
    N=2048 ALG=2 Threads=4
      Tiempo: 49.7716 ms
    N=2048 ALG=3 Threads=4
      Tiempo: 7.08573 ms

  CPU vs GPU
  Se obtuvo una aceleración (Speedup) aproximada de 13,000x. Esto ocurre porque la multiplicación de matrices es una tarea altamente paralelizable.
  La GPU utiliza sus miles de núcleos CUDA para calcular miles de elementos de la matriz resultante simultáneamente, ocultando la latencia de acceso a memoria mediante el cambio rápido de contexto entre hilos.

  GPU basica vs GPU compartida
  La versión con memoria compartida es ~7 veces más rápida.
  
  Basica: Cada hilo accede de forma independiente a la Memoria Global (DRAM) para leer la fila de A y la columna de B completas. Esto genera un tráfico de memoria masivo y redundante, ya que el mismo dato es leído desde la memoria global múltiples veces por diferentes hilos

  Compartida: Se utiliza la técnica de Tiling (bloques). Los hilos de un bloque colaboran para cargar una pequeña sub-matriz ("tile") desde la Memoria Global a la Memoria Compartida. Una vez que los datos están en la memoria compartida, son reutilizados múltiples veces por los hilos del bloque sin tener que volver a consultar la lenta memoria global. Reduciendo drásticamente el ancho de banda necesario, permitiendo que los núcleos de la GPU operen más cerca de su capacidad máxima de cómputo.

## Graficos 
<img width="849" height="548" alt="download" src="https://github.com/user-attachments/assets/f9247310-1920-48fe-9989-364925f2803e" />

<img width="868" height="547" alt="download (1)" src="https://github.com/user-attachments/assets/c3e59a2e-dc7a-4918-8b24-06fb8d35b8f1" />

## Conclusión 
  El experimento demuestra que, aunque el paralelismo masivo de la GPU ofrece una mejora inmediata sobre la CPU, el verdadero rendimiento se desbloquea al gestionar eficientemente la jerarquía de memoria.
