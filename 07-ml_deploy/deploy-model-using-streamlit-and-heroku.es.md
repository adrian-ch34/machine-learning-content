---
description: >-
  Aprende a implementar modelos de Machine Learning con Streamlit y Heroku.
  Descubre cómo crear aplicaciones web interactivas de forma sencilla.
---
# Implementación de un modelo de Machine learning usando Streamlit y Heroku

Imagina que puedes convertir secuencias de comandos simples de Python en hermosas aplicaciones web. Bueno, esa herramienta existe y se llama Streamlit.

Streamlit es un marco de código abierto para crear aplicaciones de ciencia de datos y Machine Learning para la exploración de datos de la manera más rápida posible. Incluso te brinda una experiencia de codificación en tiempo real. Puedes iniciar tu aplicación streamlit y cada vez que guardes, ¡verás tu código reflejado en el navegador al mismo tiempo!

## Los principios básicos de Streamlit:

1. **Adopta las secuencias de comandos de Python.** Si sabes cómo escribir secuencias de comandos (*scripts*) de Python, puedes escribir aplicaciones Streamlit. Por ejemplo, así es como se escribe en la pantalla:

```py
import streamlit as st
st.write('Hello, world!')
```

![Streamlit texto simple](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/streamlit1.jpg?raw=true)

2. **Trata los widgets como variables.** ¡No hay callbacks en Streamlit! Cada interacción simplemente vuelve a ejecutar el script de arriba abajo. Este enfoque conduce a un código realmente limpio:

```py
import streamlit as st
x = st.slider('x')
st.write(x, 'squared is', x * x)
```

![Streamlit slider](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/streamlit2.jpg?raw=true)

3. **Reutiliza datos y cálculos.** ¿Qué sucede si descargas muchos datos o realizas cálculos complejos? La clave es reutilizar de forma segura la información entre ejecuciones. Streamlit presenta una funcionalidad de caché que se comporta como un almacén de datos persistente e inmutable de forma predeterminada que permite que las aplicaciones de Streamlit reutilicen la información de forma segura y sin esfuerzo. Mira el siguiente ejemplo:

```py
import streamlit as st
import pandas as pd

# ¡Reutiliza estos datos en las ejecuciones!
read_and_cache_csv = st.cache(pd.read_csv)

BUCKET = "https://streamlit-self-driving.s3-us-west-2.amazonaws.com/"
data = read_and_cache_csv(BUCKET + "labels.csv.gz", nrows=1000)
desired_label = st.selectbox('Filter to:', ['car', 'truck'])
st.write(data[data.label == desired_label])
```

![Streamlit selectbox](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/streamlit3.jpg?raw=true)

Ahora, sigamos adelante e instalemos `streamlit` usando `pip`:

```bash
pip install --upgrade streamlit
```

Una vez completada la instalación, usa el siguiente comando para ver una demostración de una aplicación con código de ejemplo:

```bash
streamlit hello
```

Ahora puedes ver tu aplicación en su navegador: `http://localhost:8501`

La simplicidad de estas ideas no te impide crear aplicaciones increíblemente ricas y útiles con Streamlit.

- Las aplicaciones Streamlit son archivos Python puros. Para que puedas usar tu editor y depurador favorito con Streamlit.

- Los scripts de Python puro funcionan a la perfección con Git y otros software de control de código fuente, incluidas confirmaciones, solicitudes de incorporación de cambios, problemas y comentarios. Debido a que el lenguaje subyacente de Streamlit es Python puro, obtiene todos los beneficios.

- Streamlit proporciona un entorno de codificación en vivo de modo inmediato. Simplemente, haz clic en `Rerun` siempre cuando Streamlit detecte un cambio en el archivo de origen.

- Streamlit está diseñado para GPU. Streamlit permite el acceso directo a primitivas a nivel de máquina como TensorFlow y PyTorch y complementa estas librerías.

![Streamlit ejemplo de código](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/streamlit4.jpg?raw=true)

## Características básicas de Streamlit 

Aquí explicaremos algunas de las características básicas, pero para obtener una documentación completa de Streamlit, puedes hacer clic en el siguiente enlace: `https://docs.streamlit.io/`

**Widgets de selección**

Hay muchos widgets disponibles, incluidos los siguientes:

### SelectBox

```py
age = streamlit.selectbox("Choose your age:", np.arange(18, 66, 1))
```

![Streamlit selectbox](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/streamlit_selectbox.jpg?raw=true)

Otra opción:

```py
select = st.selectbox('Select a State', data['State'])
```

El primer parámetro es el título del cuadro de selección y el segundo parámetro define una lista de valores que se completarán en el cuadro de selección. En el segundo ejemplo hay una columna "State" del archivo `.csv` que cargamos.

### Slider

```py
age = streamlit.slider("Choose your age: ", min_value=16, max_value=66, value=35, step=1)
```

![Streamlit slider](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/streamlit_slider.jpg?raw=true)

### Multiselect

```py
artists = st.multiselect("Who are your favorite artists?", 
                         ["Michael Jackson", "Elvis Presley",
                         "Eminem", "Billy Joel", "Madonna"])
```

![Streamlit multiselect](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/streamlit_multiselect.jpg?raw=true)

### Checkbox

```py
st.sidebar.checkbox("Show Analysis by State", True, key=1)
```

El primer parámetro en la casilla de verificación define el título de la casilla de verificación, el segundo parámetro define *True* o *False* si está marcada de forma predeterminada o no y el tercer parámetro define la clave única para la casilla de verificación.

## Almacenamiento en caché

El problema con muchas herramientas de dashboard es que los datos se vuelven a cargar cada vez que seleccionas una opción o cambia de página. Afortunadamente, Streamlit tiene una opción increíble que te permite almacenar en caché los datos y solo ejecutarlos si no se han ejecutado antes. Puedes almacenar en caché cualquier función que crees. Esto puede incluir cargar datos, pero también preprocesarlos o entrenar un modelo complejo una vez.

```py
import pandas as pd
import streamlit as st

@st.cache
def load_data():
    df = pd.read_csv("your_data.csv")
    return df

# Solo se ejecutará una vez si ya está en caché
df = load_data()
```

## Visualización

Streamlit admite muchas librerías de visualización, incluidas: Matplotlib, Altair, Vega-Lite, Plotly, Bokeh, Deck.GL y Graphviz. ¡Incluso puede cargar audio y video!

```py
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st

df = pd.DataFrame(np.random.randn(200, 3), columns=['a', 'b', 'c'])
c = alt.Chart(df).mark_circle().encode(x='a', y='b', size='c',  
                                       color='c')
st.altair_chart(c, width=-1)
```

![Streamlit visualización](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/streamlit_visualization.jpg?raw=true)

Un ejemplo diferente:

```py
def get_total_dataframe(dataset):
    total_dataframe = pd.DataFrame({
    'Status':['Confirmed', 'Active', 'Recovered', 'Deaths'],
    'Number of cases':(dataset.iloc[0]['Confirmed'],
    dataset.iloc[0]['Active'], dataset.iloc[0]['Recovered'],
    dataset.iloc[0]['Deaths'])})
    return total_dataframe
state_total = get_total_dataframe(state_data)
if st.sidebar.checkbox("Show Analysis by State", True, key=2):
    st.markdown("## **State level analysis**")
    st.markdown("### Overall Confirmed, Active, Recovered and " +
    "Deceased cases in %s yet" % (select))
    if not st.checkbox('Hide Graph', False, key=1):
        state_total_graph = px.bar(
        state_total, 
        x='Status',
        y='Number of cases',
        labels={'Number of cases':'Number of cases in %s' % (select)},
        color='Status')
        st.plotly_chart(state_total_graph)
```

![Streamlit visualización 2](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/streamlit_visualization2.jpg?raw=true)

Para trazar el gráfico, usamos el método de barras de la librería `plotly.express`. El primer parámetro es el marco de datos que queremos trazar, el segundo parámetro es la columna del eje x, el tercer parámetro es la columna del eje y, el parámetro de etiquetas es opcional en caso de que desees cambiar el nombre de una columna para el gráfico, y el parámetro de color aquí es para codificar por color el gráfico sobre la base de la columna Estado del marco de datos.

## Markdown

Podemos generar Markdown y hermosos README con una sola función:

```py
import streamlit as st
st.markdown("### 🎲 The Application")
st.markdown("This application is a Streamlit dashboard hosted on Heroku that can be used "
            "to explore the results from board game matches that I tracked over the last year.")
st.markdown("**♟ General Statistics ♟**")
st.markdown("* This gives a general overview of the data including "
            " frequency of games over time, most games played in a day, and longest break "
            "between games.")
```

![Streamlit markdown](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/streamlit_markdown.jpg?raw=true)

## Función Write

La función `write` se comporta de manera diferente en función de su entrada. Por ejemplo, si agregas una figura de Matplotlib, automáticamente te mostrará esa visualización.

Algunos ejemplos:

```py
write(string) : Prints the formatted Markdown string.
write(data_frame) : Displays the DataFrame as a table.
write(dict) : Displays dictionary in an interactive widget.
write(keras) : Displays a Keras model.
write(plotly_fig) : Displays a Plotly figure.
```

## Creando la aplicación

Veamos cómo podemos crear una aplicación web de ejemplo muy básica. Primero crearemos un archivo Python llamado `app.py` e importaremos las librerías que necesitaremos.

```py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
```

Luego importamos los datos:

```py
@st.cache(ttl=60*5, max_entries=20)
def load_data():
    data = pd.read_csv('https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/titanic_train.csv')
    return data

data = load_data()
```

En el método `load_data()`, estamos leyendo el archivo `.csv` usando la librería Pandas y estamos haciendo que nuestro código sea eficiente almacenando en caché los datos. Si estos datos siguieran cambiando, borramos nuestra memoria caché cada 5 minutos o para un máximo de 20 entradas. Si los datos no cambian con mucha frecuencia, simplemente podemos usar `@st.cache(persist=True)`. El código anterior es un ejemplo, pero para el modelo Titanic, podríamos mantener `persist=True`.

Ahora vamos a crear un título, algo de contenido y un menú lateral.

```py
st.markdown('<style>description{color:blue;}</style>', unsafe_allow_html=True)
st.title('Titanic survival prediction')
st.markdown("<description>The sinking of the Titanic is one of the most infamous shipwrecks in history. " + 
"On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding " +
"with an iceberg. Unfortunately, there weren't enough lifeboats for everyone onboard, resulting in the death of " +
"1502 out of 2224 passengers and crew. While there was some element of luck involved in surviving, it seems some " +
"groups of people were more likely to survive than others. </description>", unsafe_allow_html=True)
st.sidebar.title('Select the parameters to analyze survival prediction')
```

La descripción se muestra en color azul porque usamos HTML para dar el color personalizado como azul. También podemos usar encabezado y subencabezado como usamos `st.title()` para diferentes encabezados. O podemos usar Markdown para ese propósito.

Cualquier cosa que llamemos en la barra lateral se mostrará en ella.

Una vez que hayas terminado de crear tu propia aplicación, puedes ejecutarla usando:

```bash
streamlit run app.py
```

## Implementación

Ahora que tenemos una aplicación web muy básica, podemos mostrársela a otros al implementarla en Heroku. Por supuesto, Heroku no es la única opción gratuita en el mercado. Una opción gratuita diferente podría ser Azure, Render, Amazon EC2 y muchas otras.

Si ya habías instalado la interfaz de línea de comandos (CLI) de Heroku, entonces estás listo para comenzar. Si no, puedes hacerlo desde aquí: 
https://devcenter.heroku.com/articles/getting-started-with-python#set-up

Esto lo ayudará a administrar su aplicación, ejecutarla localmente, ver sus registros y mucho más.

## Proceso de implementación

- Abre tu `cmd.exe` e ingresa a la carpeta de la aplicación.

- Inicia sesión en Heroku con `heroku login`. Será redirigido a una pantalla de inicio de sesión en tu navegador preferido.

- Mientras tienes tu cmd abierto en la carpeta de su aplicación, primero ejecuta `heroku create` para crear una instancia de Heroku.

- Hazle push a todo tu código a esa instancia con `git push heroku main`.

Esto creará una instancia de Heroku y enviará todo el código de la carpeta de la aplicación a esa instancia. Ahora, la aplicación debe implementarse.

- Con `heroku ps:scale web=1` se asegurará de que se esté ejecutando al menos una instancia de la aplicación.

- Finalmente, ejecuta `heroku open` para abrir tu aplicación en el navegador.


Fuente:

https://docs.streamlit.io/

https://www.heroku.com/

https://medium.com/towards-data-science/streamlit-101-an-in-depth-introduction-fc8aad9492f2

https://medium.com/insiderfinance/python-streamlit-app-lets-you-get-any-stock-information-with-just-4-lines-of-code-128b784afab8

https://medium.com/towards-data-science/quickly-build-and-deploy-an-application-with-streamlit-988ca08c7e83

https://medium.com/dataseries/interactive-convolutional-neural-network-65bc19d8d698

https://medium.com/towards-data-science/how-to-deploy-a-streamlit-app-using-an-amazon-free-ec2-instance-416a41f69dc3

https://medium.com/swlh/using-streamlit-to-create-interactive-webapps-from-simple-python-scripts-f78b08e486e7

https://towardsdatascience.com/coding-ml-tools-like-you-code-ml-models-ddba3357eace

https://neptune.ai/blog/streamlit-guide-machine-learning
