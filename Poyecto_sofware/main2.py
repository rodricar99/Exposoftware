import pandas as pd
import openai
import matplotlib.pyplot as plt
import time

# Configurar la API key de OpenAI
openai.api_key = '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'

# Definir parámetros del Circuit Breaker
max_retries = 3
retry_delay = 1
circuit_breaker_timeout = 10

# Crear instancia del Circuit Breaker
class CircuitBreaker:
    def __init__(self):
        self.state = 'CLOSED'
        self.last_failure_time = None

    def execute(self, func):
        if self.state == 'OPEN':
            if self.last_failure_time is not None and time.time() - self.last_failure_time >= circuit_breaker_timeout:
                self.state = 'HALF-OPEN'
            else:
                raise Exception('Circuit Breaker is OPEN')

        try:
            result = func()
            self.state = 'CLOSED'
            return result
        except Exception as e:
            self.state = 'OPEN'
            self.last_failure_time = time.time()
            raise e

# Crear instancia del Circuit Breaker
circuit_breaker = CircuitBreaker()

# Cargar los datos desde el archivo CSV
def load_data():
    return pd.read_csv('world_cup_matches.csv')

data = circuit_breaker.execute(load_data)

# Filtrar los datos para incluir solo los partidos en los que el equipo de casa ha ganado
victorias_data = data[data['Home Goals'] > data['Away Goals']]

# Obtener los equipos únicos que han tenido victorias
equipos_victorias = victorias_data['Home Team'].unique()

# Calcular el número de victorias por equipo en los datos históricos
victorias = victorias_data['Winning Team'].value_counts().to_dict()

# Calcular el total de victorias
total_victorias = sum(victorias.values())

# Calcular el número de partidos ganados por cada equipo en los datos históricos
partidos_ganados = data['Winning Team'].value_counts().to_dict()

# Calcular el total de partidos ganados por todos los equipos
total_partidos_ganados = sum(partidos_ganados.values())

# Generar las probabilidades de ser campeón según la perspectiva de OpenAI para cada equipo
def get_openai_probability(equipo):
    prompt = f"¿Cuál es la probabilidad de que {equipo} sea campeón del Mundial?"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=10,
        n=1,
        stop=None
    )
    probability = response.choices[0].text.strip()
    return probability

probabilidades_openai = []
for equipo in equipos_victorias:
    probability = circuit_breaker.execute(lambda: get_openai_probability(equipo))
    probabilidades_openai.append((equipo, probability))

# Combinar las probabilidades históricas y las probabilidades de OpenAI
probabilidades_combinadas = []

for equipo, probabilidad_openai in probabilidades_openai:
    victorias_equipo = victorias.get(equipo, 0)
    probabilidad_historica = victorias_equipo / total_victorias

    partidos_ganados_equipo = partidos_ganados.get(equipo, 0)
    probabilidad_partidos_ganados = partidos_ganados_equipo / total_partidos_ganados

    try:
        probabilidad_openai = float(probabilidad_openai)
    except ValueError:
        probabilidad_openai = 0.0

    # Aumentar la probabilidad si el equipo llegó y ganó la final
    if equipo in data[data['Stage'] == 'Final']['Winning Team'].unique():
        probabilidad_combinada = (probabilidad_historica + probabilidad_openai + probabilidad_partidos_ganados + 0.1) / 4
    else:
        probabilidad_combinada = (probabilidad_historica + probabilidad_openai + probabilidad_partidos_ganados) / 3

    probabilidades_combinadas.append((equipo, probabilidad_combinada))

# Crear un DataFrame con las probabilidades combinadas
df_probabilidades = pd.DataFrame(probabilidades_combinadas, columns=['Equipo', 'Probabilidad'])

# Ordenar los equipos por su probabilidad de ser campeón
df_probabilidades = df_probabilidades.sort_values('Probabilidad', ascending=False)

# Obtener los primeros 10 y últimos 10 equipos
primeros_10_equipos = df_probabilidades.head(10)
ultimos_10_equipos = df_probabilidades.tail(10)

# Concatenar los primeros 10 y últimos 10 equipos
equipos_seleccionados = pd.concat([primeros_10_equipos, ultimos_10_equipos])

# Crear el gráfico de barras con los equipos seleccionados
plt.figure(figsize=(12, 6))
plt.bar(equipos_seleccionados['Equipo'], equipos_seleccionados['Probabilidad'])
plt.xlabel('Equipo')
plt.ylabel('Probabilidad')
plt.title('Probabilidades de ser campeón del Mundial (10 primeros y 10 últimos)')

# Rotar los nombres de los equipos en el eje x en un ángulo de 90 grados
plt.xticks(rotation=90)

# Ajustar los márgenes del gráfico para evitar recorte de etiquetas
plt.subplots_adjust(bottom=0.4)

# Mostrar el gráfico
plt.show()
