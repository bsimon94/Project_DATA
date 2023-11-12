"""
Created on Thu August 12 13:29:18 2023
@author: Simon jules Bakoa & Nathalie DI STEFANO
"""
# Dans le fichier Video_Games_Sales.py, importer la librairie Streamlit et les librairies d'exploration de données et de DataVizualization nécessaires.
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from scipy import stats
from PIL import Image

# Création des Dataframe
# DataFrame de départ
df_origine = pd.read_csv('vgsales.csv')
# DataFrame final
df = pd.read_csv("vgsales_enrichi.csv")

# DataFrame après le webscraping
df_meta = pd.read_csv('vgsales_enrichi_metacritic.csv')
df_autre = pd.read_csv('vgsales_enrichi_autres.csv')
df_web = df_meta.merge(df_autre, on = 'Rank', how = 'outer')
df_web = df_web.drop(['Name_y', 'Platform_y', 'Year_y', 'Genre_y', 'Publisher_y', 'NA_Sales_y', 'EU_Sales_y', 'JP_Sales_y', 'Other_Sales_y', 'Global_Sales_y', 'Platform_rename'], axis = 1)
df_web = df_web.rename(columns = {'Name_x': 'Name', 'Platform_x': 'Platform', 'Year_x': 'Year', 'Genre_x': 'Genre', 'Publisher_x': 'Publisher', 'NA_Sales_x': 'NA_Sales', 'EU_Sales_x': 'EU_Sales', 'JP_Sales_x': 'JP_Sales', 'Other_Sales_x': 'Other_Sales', 'Global_Sales_x': 'Global_Sales'})

# Copier le code suivant pour ajouter un titre au Streamlit et créer 7 pages appelées "Introduction", "Webscraping", "Consolidation", "Exploration", "DataVizualization", "Pré-processing", "Modélisation".
st.title("Analyse et Prédiction des ventes de jeux vidéo")

st.write("___")

st.sidebar.title("Sommaire")
pages=["Introduction", "Webscraping", "Consolidation", "Exploration", "DataVisualization", "Pré-processing", "Modélisation"]
page=st.sidebar.radio("Aller vers", pages)

if page == pages[0]:

  # Titre de la page
  st.header("Introduction")
  
  # Première partie
  st.subheader("1. Objectif du projet")
  st.write(":point_right: prédire les ventes totales d’un jeu vidéo à l’aide d’informations descriptives comme **Le studio l’ayant développé**, **L’éditeur l’ayant publié**, **La plateforme sur laquelle le jeu est sortie**, **Le genre**, **l'avis des experts ayant testé le jeu avant sa sortie** et **la note des utilisateurs**")
  
  # Deuxième partie
  st.subheader("2. Présentation des données de départ")
  # Données d'origine
  pre, der = st.tabs(["10 premières lignes", "10 dernières lignes"])
  with pre:
    st.write("Echantillon des 10 premières lignes :", df_origine.head(10))
  with der:
    st.write("Echantillon des 10 dernières lignes :", df_origine.tail(10))
  # Taille du Dataframe
  st.write("La taille du DataFrame :", df_origine.shape)
  # Précision sur différents champs
  selection = st.selectbox(label = "Sélectionnez un champ", options = ["Platform", "Year", "Genre", "Publisher", "Sales"])
  if selection == 'Platform':
    st.write("Les valeurs :", sorted(df_origine.Platform.unique()))
  if selection == 'Year':
    st.write("Année la plus vieille :", df_origine.Year.min())
    st.write("Année la plus récente :", df_origine.Year.max())
  if selection == 'Genre':
    st.write("Les valeurs :", sorted(df_origine.Genre.unique()))
  if selection == 'Publisher':
    st.write("Nombre total d'éditeurs :", len(df_origine.Publisher.unique()))
    st.write("Le Top 10 éditeurs :", df_origine.Publisher.value_counts().head(10))
  if selection == 'Sales':
    st.write("NA_Sales = ventes en Amérique du Nord (par millions)")
    st.write("EU_Sales = ventes en Europe (par millions)")
    st.write("JP_Sales = ventes au Japon (par millions)")
    st.write("Other_Sales = ventes dans les autres pays (par millions)")
    st.write("Global_Sales = ventes dans le monde (par millions) et donc la somme des autres colonnes")
  # Valeurs manquantes
  if st.checkbox("Afficher les valeurs manquantes"): 
    st.dataframe(df_origine.isna().sum())
    st.write("Nombre de lignes avec l'éditeur à 'Unknown' :", len(df_origine[df_origine.Publisher == 'Unknown']))
  
  # Troisième partie
  st.subheader("3. Conclusion de cette introduction")
  st.write(":point_right: Récupérer des données supplémentaires sur internet.")

elif page == pages[1]: 
  
  # Titre de la page
  st.header("Webscraping")

  # Première partie
  st.subheader("1. Quelles données récupérer ? :memo:")
  st.markdown("- Le développeur \n - La description des jeux \n - La note professionnelle \n - La note moyenne des joueurs \n - La classification \n - Le mode de jeu")
  
  # Deuxième partie
  st.subheader("2. Quels sites ? :spider_web:")
  st.write("- metacritic.com")
  st.image(Image.open('metacritic.png'))
  st.write("- jeuxvideo.com")
  st.image(Image.open('jeuxvideocom.png'))
  st.write("- wikipedia.org")
  st.image(Image.open('wikipedia.png'))
  st.write("- google.com")
  st.image(Image.open('google.png'))    
  
  # Troisième partie
  st.subheader("3. Les problèmes rencontrés :bone:")
  st.write("- Les plantages des sites \n - La ponctuation, les majuscules dans les titres \n - Les titres non exacts \n - Les données pas écrites de la même manière sur wikipedia \n - Les temps de récupération assez longs")
  
  # Quatrième partie
  st.subheader("4. Résultat :heavy_check_mark:")
  st.dataframe(df_web.head(10))

elif page == pages[2]: 
     
  # Titre de la page
  st.header("Consolidation des données récupérées")
  
  # Publisher
  st.subheader("Étape 1 : Publisher")
  st.write("- Remplacement des 'Unknown' par NaN \n - Remplacement des valeurs manquantes par la colonne 'Editeur'")
  
  # Year
  st.subheader("Étape 2 : Year")
  st.write("- Remplacement des valeurs erronées ('nnue', 'nulé') dans la colonne 'Annee_Sortie' par NaN \n - Changement de son type par 'float' \n - Remplacement des valeurs manquantes de 'Year' par la colonne 'Annee_Sortie'")
  
  # Developpeur
  st.subheader("Étape 3 : Developpeur")
  st.write("- Création d'une colonne 'Developpeur' avec 'Developpeur_M' \n - Remplacement des valeurs manquantes par la colonne 'Developpeur_W'")
  
  # Notes
  st.subheader("Étape 4 : Les notes")
  st.write("- Passage des notes sur 100 (les notes pro de Metacritic sont sur 100) \n - Rassemblement des notes pro Metacritic et JVC dans une nouvelle variable 'NotePro' \n - Rassemblement des notes des joueurs Metacritic et JVC dans une nouvelle variable 'NoteJoueurs'")
  
  # Solo et Online
  st.subheader("Étape 5 : Ajout des colonnes 'Solo' et 'Online'")
  st.write("- Utilisation des colonnes 'NBJ_Online', 'Nb_Joueurs', 'Mode_Jeu'")
  col1, col2, col3 = st.tabs(["NBJ_Online", "Nb_Joueurs", "Mode_Jeu"])
  with col1:
      st.write("Valeurs de NBJ_Online :", df_web.NBJ_Online.unique())
  with col2:
      st.write("Valeurs de Nb_Joueurs :", df_web.Nb_Joueurs.unique())
  with col3:
      st.write("Valeurs de Mode_Jeu :", df_web.Mode_Jeu.unique())
  
  # Résultat final
  st.subheader("Résultat final :heavy_check_mark:")
  st.dataframe(df.head(10))

elif page == pages[3]:

  # Titre de la page
  st.header("Exploration et nettoyage des données")

  # Affichage des données du Dataframe
  st.write("10 premières lignes :", df.head(10))
  # Taille des données
  st.write("Taille du DataFrame :", df.shape)
  
  st.subheader("Valeurs manquantes")
  # Créer une checkbox pour choisir d'afficher ou non le nombre de valeurs manquantes en utilisant la méthode st.checkbox()
  if st.checkbox("Afficher les valeurs manquantes") :
    st.dataframe(df.isna().sum())
  
  NA1, NA2, NA3, NA4 = st.tabs(["Year", "Publisher", "Rating", "Notes"])
  # Year
  with NA1:
    st.write("**Year :**", df[df.Year.isna()])
    st.write("Jeu sans information et un jeu annulé :point_right: Suppression de ces deux lignes")
  # Publisher
  with NA2:
    st.write("**Publisher :**", df[df.Publisher.isna()])
    st.write("Ce sont des films sortis sur GBA :point_right: Suppression de ces lignes")
  # Rating
  with NA3:
    st.write("**Rating :**")
    st.write(df.Rating.isna().sum(), " valeurs manquantes :point_right: Choix d'abandonner cette colonne")
  # Notes
  with NA4:
    st.write("**NotePro et  NoteJoueurs :**")
    # Boxplots des notes
    df_notes = df.dropna(subset = ['NotePro','NoteJoueurs'])
    fig_notes = plt.figure()
    plt.boxplot([df_notes.NotePro, df_notes.NoteJoueurs]) 
    plt.title("Boxplots des notes")
    st.pyplot(fig_notes)
    st.write("- Moyenne des notes pro :", df.NotePro.mean(), "\n - Médiane des notes pro :", df.NotePro.median())
    st.write("- Moyenne des notes joueurs :", df.NoteJoueurs.mean(), "\n - Médiane des notes joueurs :", df.NoteJoueurs.median())
    st.write(" :point_right: Remplacement des valeurs manquantes par la moyenne")
  
  # Descriptions
  st.subheader("Nuages de mots avec les descriptions")
  st.write("**Description_M :**")
  st.write("Nombre de valeurs manquantes :", df.Description_M.isna().sum())
  st.image(Image.open('Nuage_Mots_Desc_Anglais.png'))
  st.write("**Description_JVC :**")
  st.write("Nombre de valeurs manquantes :", df.Description_JVC.isna().sum())
  st.image(Image.open('Nuage_Mots_Desc_Francais.png'))
  st.write("Trop peu de mots significatifs en ressortent. Les mots 'combat', 'monde', 'action', 'ami' ou 'course' apparaissent, mais ils peuvent s'appliquer à tous les jeux.")
  st.write("Il aurait fallu récupérer les commentaires des joueurs plutôt que les descriptions qui peuvent être assez génériques.")
  st.write(" :point_right: Choix d'écarter ces variables")
  
  # Solo et Online
  st.subheader("Colonnes 'Solo' et 'Online'")
  st.write(df.Solo.isna().sum(), " valeurs manquantes :point_right: Choix d'abandonner ces deux colonnes")
  # Répartitions
  st.write("Distribution de ces deux variables :")
  var1, var2 = st.tabs(["Solo", "Online"])
  # Solo
  with var1:
    st.write("Nombre de jeux Solo (1) ou non (0) :", df.Solo.value_counts())
    st.write("Une bien plus grande part de jeux multijoueurs")
  # Publisher
  with var2:
    st.write("Nombre de jeux Online (1) ou non (0) :", df.Online.value_counts())
    st.write("Beaucoup de vieux jeux quand internet n'était pas répandu peut expliquer le nombre important de jeux offline")

elif page == pages[4]:

  # Ecrire "DataVizualization" en haut de la deuxième page en utilisant la commande st.write() dans le script Python.
  st.header("DataVisualization")

  # Conversion en type entier les variables 'year', 'Online' et 'Solo' qui sont de type décimal.
  df["Year"] =  df["Year"].astype("Int64")
  df["Online"] =  df["Online"].astype("Int64")
  df["Solo"] =  df["Solo"].astype("Int64")

  # definition du périmètre du jeu de donnée
  df = df.loc[(df["Year"] >= 1980) & (df["Year"] <= 2016)]

  # Suppression des lignes dont les champs Year et Publisher ont des valeurs manquantes
  df = df.dropna(subset = 'Year')
  # Suppression des lignes dans Publisher : films sortis sur GBA
  df = df.dropna(subset = 'Publisher')
  
  # Afficher la somme des ventes totales (à l'echelle mondiale) pour les 5 meilleurs éditeurs de jeux vidéo
  st.write(":point_right: la somme des ventes totales (à l'echelle mondiale) pour les 5 meilleurs éditeurs de jeux vidéo")
  top5Publisher = df.copy()
  top5Publisher = top5Publisher.groupby(by="Publisher")["Global_Sales"].sum().reset_index()
  top5Publisher = top5Publisher.sort_values('Global_Sales', ascending=False).reset_index(drop=True)
  st.dataframe(top5Publisher.head())
  
  # Nous allons afficher pour chacun des 5 meilleurs éditeurs 'Nintendo', 'Electronic Arts', 'Activision', 'Sony Computer Entertainment' et 'Ubisoft'
  # sur un graphique à barres horizontal (Barplot) son volume de ventes globales par année
  # sur un camenbert (Pie plot) son pourcentage de ventes globales par le genre du jeu
  # sur un viollon (violin plot) son volume de ventes globales par le genre du jeu
  st.write(":point_right: afficher pour chacun des 5 meilleurs éditeurs **Nintendo**, **Electronic Arts**, **Activision**, **Sony Computer Entertainment** et **Ubisoft**")
  # Placing the names of the top 5 publishers into an array
  top5 = ["Nintendo", "Electronic Arts", "Activision", "Sony Computer Entertainment", "Ubisoft"]
  # Selecting a colour palette
  pal = sns.color_palette("Paired", 12)
  pal = pal.as_hex()[0:12]

  # Initializing a for loop to generate each visualization set for each publisher in the top 5 array
  for i in top5:
    # Querying for specifically one pulisher
    publisherData = df.query(f"Publisher == '{i}'")
    fig = plt.figure()

    # Initializing a subplot layout of 3 figures
    gs = fig.add_gridspec(2,2)
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[:, 0])
    fig.suptitle(f'Visualization of distribution of sales ({i})')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95], pad=1)

    # Generating a Violin plot of Global sales by Genre
    try:
      sns.violinplot(x="Genre", y="Global_Sales", data=publisherData,ax=ax2, scale="width", inner=None, linewidth=1, edgecolor="black", palette=pal)
      ax2.tick_params(labelrotation=45, axis="x", labelsize=8)
      plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
      ax2.set_ylabel('Global Sales (Per Million)', fontsize=8)
      ax2.set_title('Global Sales', fontsize=8, fontweight="bold")
      ax2.set_xlabel('', fontsize=8)
    except:
      pass

    # Generating a Horizontal Bar plot of the Global sales by year
    try:
      publisherData["Year"] = publisherData["Year"].astype("category")
      sns.barplot(y="Year", x="Global_Sales", data=publisherData, ax=ax3, errorbar=None, palette="ch:.25", linewidth=0.5, edgecolor="black", width=1, orient="h")
      ax3.set_ylabel('Year', fontsize=10)
      ax3.set_xlabel('Global Sales (Per Million)')
      ax3.tick_params(axis="y", labelsize=8)
      ax3.set_title('Sales By Year', fontsize=10, fontweight="bold")
    except:
      pass

    # Generating a pie chart of global sales by genre
    try:
      genregroup = publisherData.groupby(['Genre'])["Global_Sales"].sum()
      genregroup.plot(kind='pie', y=["Global_Sales"], ax=ax1, legend=True,labeldistance=None, wedgeprops={'edgecolor':'black','linewidth': 1, 'linestyle': 'solid'}, colors=pal)
      ax1.legend(bbox_to_anchor=(1.68,1.2), loc="upper right", prop={"size":8}, frameon=False)
      box = ax1.get_position()
      box.x0 = box.x0 - 0.08
      box.x1 = box.x1 - 0.08
      ax1.set_position(box)
      ax1.set_ylabel('% of Sales By Genre', fontsize=10)
    except:
      pass
    st.pyplot(fig)

  # tracer des diagrammes de régression linéaire et calculer les valeurs R2 et P pour la corrélation entre la variable cible 'Global_Sales' et la variablle 'EU_Sales'
  st.write(":point_right: les diagrammes de régression linéaire et calcul des valeurs R2 et P pour la corrélation entre la variable cible **Global_Sales** et la variable **EU_Sales** par genre de jeu.")
  fig, axes = plt.subplots(3, 4, figsize=(15,15))
  fig.suptitle(f'Correlation Plots of GL/EU sales by Genre, with calculated linear regression values', fontsize=20, fontweight="bold", wrap=True)
  x=0
  y=0
  z=0
  # Finding and saving a list of all unique Genres
  genres = df["Genre"].unique()
  for i in genres:
    genreData = df.query(f"Genre == '{i}'")
    sns.regplot(data=genreData, ax=axes[y,x], x="EU_Sales", y="Global_Sales", color=pal[z])
    axes[y,x].set_title(i, fontsize=18)
    datax = genreData["Global_Sales"]
    datay = genreData["EU_Sales"]
    res = stats.linregress(datax, datay)
    axes[y,x].text(0.05,0.925, f"R\u00b2: {res.rvalue**2:.2f}", transform=axes[y,x].transAxes)
    axes[y,x].text(0.05,0.875, f" P: {res.pvalue**2:.2f}", transform=axes[y,x].transAxes)

    if x==0:
      axes[y,x].set_ylabel('GL Sales')
    else:
      axes[y,x].set_ylabel('')
    if y==2:
      axes[y,x].set_xlabel('EU Sales')
    else:
      axes[y,x].set_xlabel('')
    x=x+1
    z=z+1
    if x==4:
      x=0
      y=y+1
  st.pyplot(fig)  
  
  # tracer des diagrammes de régression linéaire et calculer les valeurs R2 et P pour la corrélation entre les variables 'Global_Sales' et 'NA_Sales'
  st.write(":point_right: les diagrammes de régression linéaire et calcul des valeurs R2 et P pour la corrélation entre la variable cible **Global_Sales** et la variable **NA_Sales** par genre de jeu.")
  fig, axes = plt.subplots(3, 4, figsize=(15,15))
  fig.suptitle(f'Correlation Plots of GL/NA sales by Genre, with calculated linear regression values', fontsize=20, fontweight="bold", wrap=True)
  x=0
  y=0
  z=0
  # Finding and saving a list of all unique Genres
  genres = df["Genre"].unique()
  for i in genres:
    genreData = df.query(f"Genre == '{i}'")
    sns.regplot(data=genreData, ax=axes[y,x], x="NA_Sales", y="Global_Sales", color=pal[z])
    axes[y,x].set_title(i, fontsize=18)
    datax = genreData["Global_Sales"]
    datay = genreData["NA_Sales"]
    res = stats.linregress(datax, datay)
    axes[y,x].text(0.05,0.925, f"R\u00b2: {res.rvalue**2:.2f}", transform=axes[y,x].transAxes)
    axes[y,x].text(0.05,0.875, f" P: {res.pvalue**2:.2f}", transform=axes[y,x].transAxes)

    if x==0:
      axes[y,x].set_ylabel('GL Sales')
    else:
      axes[y,x].set_ylabel('')
    if y==2:
      axes[y,x].set_xlabel('NA Sales')
    else:
      axes[y,x].set_xlabel('')
    x=x+1
    z=z+1
    if x==4:
      x=0
      y=y+1
  st.pyplot(fig)

elif page == pages[5]: 

  # Ecrire "Pré-processing" en haut de la troisième page en utilisant la commande st.write() dans le script Python
  st.header("Pré-processing")
  
  # Import des modules necessaires
  from sklearn.model_selection import train_test_split
  from sklearn.preprocessing import StandardScaler
  from sklearn.preprocessing import LabelEncoder
  from sklearn.metrics import f1_score
  from sklearn.compose import ColumnTransformer
  from sklearn.pipeline import Pipeline
  
  # Transformation des variables catégorielles : "Platform", "Genre", "Publisher" et "Developpeur"
  st.subheader("Etape 1 : Transformation des Variables catégorielles")
  st.write(" :point_right: Nous avons encodé les 10 modalités les plus présentes des variables **Publisher** et **Developpeur** par les nombres de 0 à 9 grace à la fonction **encod(ma_variable)**")
  st.write(" :point_right: Nous avons encodé le reste des modalités et les valeurs manquantes des deux variables par le nombre 10 toujours grace à la fonction **encod(ma_variable)**")
  st.write(" :point_right: Nous avons encodé les variables **Platform** et **Genre** grace à la fonction **Pandas.get_dummies()** de Pandas")
  


  # Supprimez les variables non-pertinentes "Rank", "Rating", "Description_M", "Description_JVC", "Online", "Solo" et "Name" du DataFrame df.
  st.write(" :point_right: Nous avons supprimé les variables non-pertinentes **Rank**, **Rating**, **Description_M**, **Description_JVC**, **Online**, **Solo** et **Name**")
  df.drop(["Rank", "Name", "Developpeur", "Publisher", "NA_Sales" , "EU_Sales", "JP_Sales", "Other_Sales", "Rating", "Description_M", "Description_JVC", "Online", "Solo"], axis=1, inplace=True)
  
  # Transformation des variables numériques : 'Year', 'NotePro', 'NoteJoueurs', 'Top10Developer', 'Top10Publisher'
  st.subheader("Etape 2 : Transformation des Variables numériques")
  st.write(" :point_right: Nous avons mis à l'échelle des Variables numériques **Year**, **NotePro**, **NoteJoueurs**, **Top10Developer** et **Top10Publisher** en leur appliquant un **standardScaler()**")
  st.write(" :point_right: Nous avons remplacé les valeurs manquantes des variables numériques **NotePro** et **NoteJoueurs** par leur moyenne")
  
  # afficher le Dataframe '"preprocessed.csv"
  df_prep = pd.read_csv("preprocessed.csv")


  st.write(" :point_right: pour obtenir le dataframe préprocessé ci-dessous")
  st.dataframe(df_prep.head())

elif page == pages[6]:

  # Ecrire "Modélisation" en haut de la troisième page en utilisant la commande st.write() dans le script Python
  st.header("Modélisation")
  
  from sklearn.model_selection import train_test_split
  from sklearn.ensemble import RandomForestRegressor
  from sklearn.linear_model import LinearRegression
  from sklearn.tree import DecisionTreeRegressor
  from sklearn.metrics import r2_score
  import joblib
  
  # afficher le Dataframe '"preprocessed.csv"
  X = pd.read_csv("preprocessed.csv")
  y = pd.read_csv("Global_Sales_preprocessed.csv")
  st.write(" :point_right: Ici Nous affichons le Dataframe préprocessé sur lequel nous avons entrainé nos modèles de regression.")
  st.dataframe(X.head())
  
  # séparer les données en un ensemble d'entrainement et un ensemble test en utilisant la fonction train_test_split du package model_selection de Scikit-Learn.
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
  
  # standardiser les valeurs numériques en utilisant la fonction StandardScaler du package Preprocessing de Scikit-Learn.
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  num = ["Year", "NotePro", "NoteJoueurs", "Top10Developer", "Top10Publisher"]
  X_train[num] = scaler.fit_transform(X_train[num])
  X_test[num] = scaler.transform(X_test[num])
  
  # charger les modèles entrainés de machine learning avec la fonction load() et la librairie joblib
  reg = joblib.load("model_reg_line")
  dt = joblib.load("model_reg_dt")
  rf = joblib.load("model_reg_rf")
  
  # stocker les prédictions faites sur les modèles entrainés dans des variables y_pred_reg, y_pred_dt et y_pred_rf
  y_pred_reg=reg.predict(X_test)
  y_pred_dt=dt.predict(X_test)
  y_pred_rf=rf.predict(X_test)

  # (2) Afficher dans un graphique le nuage de points entre y_pred_reg et y_test.
  # (3) Ajouter sur ce graphique la droite d'équation  𝑦=𝑥
  import matplotlib.pyplot as plt
  
  # Afficher dans un graphique le nuage de points entre y_pred_reg et y_test.
  st.write(" :point_right: Ici Nous affichons dans un graphique le nuage de points entre **y_pred_reg** et **y_test**.")
  fig = plt.figure(figsize = (10,10))
  plt.scatter(y_pred_reg, y_test, c='green')
  plt.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()), color = 'red')
  plt.xlabel("prediction")
  plt.ylabel("vrai valeur")
  plt.title('Régression Linéaire pour la prédiction des ventes Globales de jeux Vidéo')
  plt.show()
  st.pyplot(fig)

  # utiliser la méthode st.selectbox() pour choisir un modèle de regression entre 'Linear Regression', 'Decision Tree', 'Random Forest'.
  model_choisi = st.selectbox(label = "Choix du Modèle", options = ["Linear Regression", "Decision Tree", "Random Forest"])
  st.write("Le Modèle choisi est : ", model_choisi)
  
  # créer une fonction appelée train_model qui prend en argument le model choisit et renvoie le score de prédiction du modèle choisi.
  def train_model(model_choisi):
    if model_choisi == 'Linear Regression':
      y_prep = y_pred_reg
    elif model_choisi == 'Random Forest':
      y_prep = y_pred_rf
    elif model_choisi == 'Decision Tree':
      y_prep = y_pred_dt
    r2 = r2_score(y_test, y_prep)
    return r2
  
  # affichage du score de prédiction du modèle choisi
  st.write("Le score de prédiction du modèle choisi est : ", train_model(model_choisi))
  
  # (1) Calculer ces trois métriques sur le jeu d'entraînement et le jeu de test pour le DecisionTreeRegressor,
  # (2) Calculer ces trois métriques sur le jeu d'entraînement et le jeu de test pour le RandomForestRegressor,
  # (3) Stocker l'ensemble de ces mesures dans un DataFrame et l'afficher.
  st.write(":point_right: Ici, nous affichons dans un Dataframe les trois métriques **MAE**, **MSE** et **RMSE** sur le jeu d'entraînement et le jeu de test pour les deux modèles **DecisionTreeRegressor** et **RandomForestRegressor**.")
  import sklearn.metrics
  from sklearn.metrics import mean_squared_error
  from sklearn.metrics import mean_absolute_error

  ### DecisionTree

  regressor_decision_tree = DecisionTreeRegressor(random_state=123)
  regressor_decision_tree.fit(X_train, y_train)


  y_pred_decision_tree = regressor_decision_tree.predict(X_test)
  y_pred_train_decision_tree = regressor_decision_tree.predict(X_train)

  # Calcul des métriques

  # jeu d'entraînement
  mae_decision_tree_train = mean_absolute_error(y_train,y_pred_train_decision_tree)
  mse_decision_tree_train = mean_squared_error(y_train,y_pred_train_decision_tree,squared=True)
  rmse_decision_tree_train = mean_squared_error(y_train,y_pred_train_decision_tree,squared=False)

  # jeu de test
  mae_decision_tree_test = mean_absolute_error(y_test,y_pred_decision_tree)
  mse_decision_tree_test = mean_squared_error(y_test,y_pred_decision_tree,squared=True)
  rmse_decision_tree_test = mean_squared_error(y_test,y_pred_decision_tree,squared=False)


  ### RandomForest

  regressor_random_forest = RandomForestRegressor(random_state=123)
  regressor_random_forest.fit(X_train, y_train)

  # Calcul des métriques
  y_pred_random_forest = regressor_random_forest.predict(X_test)
  y_pred_random_forest_train = regressor_random_forest.predict(X_train)


  # jeu d'entraînement
  mae_random_forest_train = mean_absolute_error(y_train,y_pred_random_forest_train)
  mse_random_forest_train = mean_squared_error(y_train,y_pred_random_forest_train,squared=True)
  rmse_random_forest_train = mean_squared_error(y_train,y_pred_random_forest_train,squared=False)

  # jeu de test
  mae_random_forest_test = mean_absolute_error(y_test,y_pred_random_forest)
  mse_random_forest_test = mean_squared_error(y_test,y_pred_random_forest,squared=True)
  rmse_random_forest_test = mean_squared_error(y_test,y_pred_random_forest,squared=False)


  # Creation d'un dataframe pour comparer les metriques des deux algorithmes
  data = {'MAE train': [mae_decision_tree_train, mae_random_forest_train],
          'MAE test': [mae_decision_tree_test, mae_random_forest_test],
          'MSE train': [mse_decision_tree_train,mse_random_forest_train],
          'MSE test': [mse_decision_tree_test,mse_random_forest_test],
          'RMSE train': [rmse_decision_tree_train, rmse_random_forest_train],
          'RMSE test': [rmse_decision_tree_test, rmse_random_forest_test]}

  # Creer DataFrame
  df = pd.DataFrame(data, index = ['Decision Tree', 'Random Forest '])
  st.dataframe(df.head())