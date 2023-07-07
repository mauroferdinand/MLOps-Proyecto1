from fastapi import FastAPI
import pandas as pd
import uvicorn
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = FastAPI()

df = pd.read_csv('nuevo_dataset.csv')

#http://127.0.0.1:8000

# Endpoint 1: Devuelve la cantidad de películas producidas en un idioma específico
@app.get('/peliculas_idioma/{idioma}')
def peliculas_idioma(idioma: str):
    cantidad_peliculas = df[df['original_language'] == idioma].shape[0]
    return {"mensaje": f"{cantidad_peliculas} cantidad de películas fueron estrenadas en idioma {idioma}"}

# Endpoint 2: Devuelve la duración y el año de una película específica
@app.get('/peliculas_duracion/{pelicula}')
def peliculas_duracion(pelicula: str):
    titulo_pelicula = df[df['title'].str.contains(pelicula, na=False)]
    duracion = titulo_pelicula['runtime'].values[0] if not titulo_pelicula.empty else None
    anio = titulo_pelicula['release_year'].values[0] if not titulo_pelicula.empty else None
    return {"mensaje": f"{pelicula}. Duración: {duracion}. Año: {anio}"}

# Endpoint 3: Devuelve la cantidad de películas, la ganancia total y el promedio de una franquicia específica
@app.get('/franquicia/{franquicia}')
def franquicia(franquicia: str):
    franquicia_data = df[df['collection_name'] == franquicia]
    cantidad_peliculas = franquicia_data.shape[0]
    ganancia_total = franquicia_data['revenue'].sum()
    ganancia_promedio = franquicia_data['revenue'].mean()
    return {"mensaje": f"La franquicia {franquicia} posee {cantidad_peliculas} películas, una ganancia total de {ganancia_total} y una ganancia promedio de {ganancia_promedio}"}

# Endpoint 4: Devuelve la cantidad de películas producidas en un país específico
@app.get('/peliculas_pais/{pais}')
def peliculas_pais(pais: str):
    cantidad_peliculas = len(df[df['production_countries_list'] == pais])
    return f"Se produjeron {cantidad_peliculas} películas en el país {pais}"


# Endpoint 5: Devuelve el revenue total y la cantidad de películas realizadas por una productora específica
@app.get('/productoras_exitosas/{productora}')
def productoras_exitosas(productora: str):
    # Filtrar el dataframe por la productora especificada
    productora_df = df[df['production_companies_list'] == productora]

    # Calcular el revenue total
    revenue_total = productora_df['revenue'].sum()

    # Obtener la cantidad de películas realizadas
    peliculas_realizadas = len(productora_df)

    # Crear el mensaje de retorno
    mensaje_retorno = f"La productora {productora} ha tenido un revenue de {revenue_total} y ha realizado {peliculas_realizadas} películas."

    return mensaje_retorno

# Endpoint 6: Información del director
@app.get('/director/{director_name}')
def buscar_director(director_name: str):
    # Filtrar el dataset por el nombre del director
    director_df = df[df['directors_name'] == director_name]
    
    if director_df.empty:
        return {"message": f"No se encontró información para el director: {director_name}"}
    
    # Obtener los datos requeridos para cada película del director
    peliculas = []
    for _, row in director_df.iterrows():
        pelicula = {
            "title": row['title'],
            "release_date": row['release_date'],
            "return": row['return'],
            "budget": row['budget'],
            "revenue": row['revenue']
        }
        peliculas.append(pelicula)
    
    # Calcular el éxito del director basado en el retorno promedio
    exito = director_df['return'].mean()
    
    # Devolver los resultados
    return {
        "director_name": director_name,
        "exito": exito,
        "peliculas": peliculas
    }
    
#SISTEMA DE RECOMENDACIÓN:

df_subset = df.head(5000)

# Creamos un vectorizador TF-IDF utilizando la columna 'overview'
tfidf = TfidfVectorizer(stop_words='english')
df_subset['overview'] = df_subset['overview'].fillna('')  # Reemplazamos los valores NaN con una cadena vacía
tfidf_matrix = tfidf.fit_transform(df_subset['overview'])

# Calculamos la similitud del coseno entre películas:
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

@app.get('/recomendacion/{titulo}')
def recomendacion(titulo: str):
    # Obtenemos el índice de la película dada
    idx = df_subset[df_subset['title'] == titulo].index[0]

    # Obtenemos las puntuaciones de similitud para todas las películas
    sim_scores = list(enumerate(cosine_similarities[idx]))

    # Ordenamos las películas según la puntuación de similitud en orden descendente
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Obtenemos los índices de las 5 películas más similares (excluyendo la película dada)
    similar_indices = [i for i, _ in sim_scores[1:6]]

    # Obtenemos los títulos de las películas similares
    similar_movies = df_subset['title'].iloc[similar_indices].values.tolist()

    return similar_movies



