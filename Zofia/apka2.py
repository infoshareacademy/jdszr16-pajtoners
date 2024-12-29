import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

def load_data():
@@ -29,15 +30,15 @@ def filter_data_by_user_id(df):
    st.session_state.user_id = st.number_input("Wpisz user_id", min_value=0, step=1, value=st.session_state.user_id)
    if st.session_state.user_id in df['user_id'].values:
        filtered_df = df[df['user_id'] == st.session_state.user_id]
        st.subheader("Podsumowanie dla wybranego użytkownika")
        st.session_state.filtered_data = filtered_df.copy()

        # Create a summary row for the selected user
        st.subheader("Podsumowanie dla wybranego użytkownika")
        summary_text = (
            f"Wiek: {filtered_df['age'].iloc[0]}\n"
            f"Płeć: {'mężczyzna' if filtered_df['gender'].iloc[0] == 1 else 'kobieta'}\n"
            f"Wzrost (cm): {filtered_df['height'].iloc[0]}\n"
            f"Waga (kg): {filtered_df['weight'].iloc[0]}\n"
            f"Suma kroków: {filtered_df['Steps'].sum():.2f}\n"
            f"Suma kroków: {filtered_df['Steps'].sum():.0f}\n"
            f"Średnie tętno: {filtered_df['Heart'].mean():.2f}\n"
            f"Suma kalorii: {filtered_df['Calories'].sum():.2f}\n"
            f"Suma dystansu (km): {filtered_df['Distance'].sum():.2f}"
@@ -52,6 +53,7 @@ def filter_data_by_user_id(df):
        return df

def plot_correlated_charts(df):
    """Function to plot the correlation charts"""
    st.header("Analiza wykresów")
    st.write("W tej sekcji prezentujemy wykresy najważniejszych korelacji między zmiennymi w zestawie danych.")
    st.subheader("Najlepiej skorelowane kolumny - Wizualizacja")
@@ -76,28 +78,19 @@ def plot_correlated_charts(df):
        ax.set_ylabel("Częstość")
        st.pyplot(fig)

def apply_pca_and_kmeans(df):
    """Apply PCA and KMeans clustering."""
    if df is not None:
        st.write("Przeprowadzamy analizę PCA i KMeans...")
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(df.select_dtypes(include=[np.number]).dropna())
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(pca_data)
        df['Cluster'] = clusters
        return pca_data, clusters
    else:
        st.warning("Najpierw wczytaj dane w zakładce 'Wczytaj dane'.")
        return None, None
def predictive_data_section():
    
    st.subheader("Dane predykcyjne")
    filtered_data = st.session_state.get('filtered_data')

    if filtered_data is not None and not filtered_data.empty:
        pca_data, clusters = apply_pca_and_kmeans(filtered_data)
        numeric_data = filtered_data.select_dtypes(include=[np.number]).dropna()
        if numeric_data.empty:
            st.warning("Brak wystarczających danych numerycznych do analizy PCA i KMeans.")
            return
        pca_data, clusters = apply_pca_and_kmeans(numeric_data)
        if pca_data is not None:
            st.write("### Wizualizacja PCA i KMeans")
            fig, ax = plt.subplots(figsize=(8, 6))
@@ -110,29 +103,52 @@ def predictive_data_section():
            ax.set_ylabel("PCA2")
            st.pyplot(fig)

            user_cluster = clusters[0]  # Zakładamy, że mamy dane tylko dla jednego użytkownika
            st.write(f"Twój klaster to: **{user_cluster + 1}**")
            cluster_descriptions = {
                0: (
                    "grupa o wysokim ryzyku",
                    "Na podstawie wysokiego poziomu cholesterolu, wysokiego ciśnienia krwi oraz niskiej aktywności fizycznej, użytkownik został przypisany do grupy wysokiego ryzyka. Zalecane działania to regularne badania medyczne, wprowadzenie diety niskotłuszczowej, zwiększenie aktywności fizycznej oraz konsultacja z lekarzem."
                ),
                1: (
                    "grupa umiarkowanego ryzyka",
                    "Na podstawie średnich wartości cholesterolu i ciśnienia krwi oraz umiarkowanej aktywności fizycznej, użytkownik został przypisany do grupy umiarkowanego ryzyka. Zalecane działania to kontynuowanie zdrowych nawyków żywieniowych, utrzymywanie aktywności fizycznej oraz regularne badania medyczne."
                ),
                2: (
                    "grupa o niskim ryzyku",
                    "Na podstawie niskiego poziomu cholesterolu, prawidłowego ciśnienia krwi oraz wysokiej aktywności fizycznej, użytkownik został przypisany do grupy niskiego ryzyka. Zalecane działania to utrzymanie zdrowego stylu życia, regularne kontrole medyczne oraz unikanie przewlekłego stresu."
                )
            }
            user_cluster = clusters[0]
            st.write(f"Nasz program przypisał Cię do klastra: **{user_cluster + 1}**")  
            
            title, description = describe_clusters(user_cluster + 1)
            st.write(f"### {title}")
            st.write(description)       
    else:
        st.warning("Najpierw wczytaj dane w zakładce 'Wczytaj dane'.")
def apply_pca_and_kmeans(df):
    """PCA and KMeans clustering."""
    if df is not None:
        st.write("Przeprowadzamy analizę PCA i KMeans...")
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(df.select_dtypes(include=[np.number]).dropna())
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(pca_data)

            cluster_description, detailed_description = cluster_descriptions.get(user_cluster, ("Nieznana grupa", "Brak dodatkowych informacji."))
            st.write(f"Opis: {cluster_description}")
            st.write(detailed_description)
        df['Cluster'] = clusters
        return pca_data, clusters
    else:
        st.warning("Najpierw wczytaj dane w zakładce 'Wczytaj dane'.")
        return None, None
def describe_clusters(cluster_id):
    cluster_descriptions = {
        0: (
            "Co oznacza klaster 0? -> Niska aktywność",
            "Użytkownicy w tym klastrze są najmniej aktywni fizycznie. Charakteryzują się niską liczbą kroków, minimalnym spalaniem kalorii i krótkimi dystansami. "
            "To może wskazywać na siedzący tryb życia lub ograniczenia fizyczne. Zaleca się stopniowe zwiększanie aktywności fizycznej, np. spacerów, "
            "oraz konsultację z lekarzem w celu oceny stanu zdrowia."
        ),
        1: (
            "Co oznacza klaster 1? -> Wysoka aktywność",
            "Osoby w tej grupie są bardzo aktywne fizycznie, regularnie osiągają wysoką liczbę kroków i dystansów. "
            "Często angażują się w ćwiczenia aerobowe lub inne formy aktywności fizycznej. Zaleca się utrzymanie obecnego poziomu aktywności oraz monitorowanie postępów, aby unikać przemęczenia."
        ),
        2: (
            "Co oznacza klaster 2? -> Umiarkowana aktywność",
            "Użytkownicy w tym klastrze charakteryzują się umiarkowanym poziomem aktywności fizycznej. "
            "Regularnie angażują się w aktywność o średniej intensywności. To zrównoważony styl życia, który warto kontynuować i rozwijać poprzez stopniowe zwiększanie intensywności ćwiczeń."
        )
    }
    
    return cluster_descriptions.get(cluster_id, ("Nieznany klaster", "Brak opisu dla tego klastra."))

def main():

@@ -184,4 +200,4 @@ def main():
        predictive_data_section()

if __name__ == "__main__":
    main()
    main()