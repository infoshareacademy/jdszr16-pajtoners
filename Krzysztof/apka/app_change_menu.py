import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

def load_data():
    """Function to load the dataset."""
    uploaded_file = st.file_uploader("", type=["csv"])
    if uploaded_file is not None:
        if "data" not in st.session_state or st.session_state.data is None:
            st.session_state.data = pd.read_csv(uploaded_file)
    return st.session_state.data

def display_data(df):
    """Function to display the loaded data."""
    st.subheader("Podgląd danych")
    if df is not None:
        st.write(df.head())

def filter_data_by_user_id(df):
    """Function to filter data by user ID."""
    if "user_id" not in st.session_state:
        st.session_state.user_id = 0

    st.session_state.user_id = st.number_input("Wpisz user_id", min_value=0, step=1, value=st.session_state.user_id)
    if st.session_state.user_id in df['user_id'].values:
        filtered_df = df[df['user_id'] == st.session_state.user_id]
        st.subheader("Podsumowanie dla wybranego użytkownika")

        # Create a summary row for the selected user
        summary_text = (
            f"Wiek: {filtered_df['age'].iloc[0]}\n"
            f"Płeć: {'mężczyzna' if filtered_df['gender'].iloc[0] == 1 else 'kobieta'}\n"
            f"Wzrost (cm): {filtered_df['height'].iloc[0]}\n"
            f"Waga (kg): {filtered_df['weight'].iloc[0]}\n"
            f"Suma kroków: {filtered_df['Steps'].sum():.2f}\n"
            f"Średnie tętno: {filtered_df['Heart'].mean():.2f}\n"
            f"Suma kalorii: {filtered_df['Calories'].sum():.2f}\n"
            f"Suma dystansu (km): {filtered_df['Distance'].sum():.2f}"
        )

        st.text_area("Szczegóły użytkownika:", value=summary_text, height=200, disabled=True)

        return filtered_df
    else:
        st.warning(f"Brak danych dla user_id: {st.session_state.user_id}")
        st.session_state.filtered_data = None
        return df

def plot_correlated_charts(df):
    st.header("Analiza wykresów")
    st.write("W tej sekcji prezentujemy wykresy najważniejszych korelacji między zmiennymi w zestawie danych.")
    st.subheader("Najlepiej skorelowane kolumny - Wizualizacja")

    correlation_matrix = df.corr(numeric_only=True)
    top_correlations = correlation_matrix.abs().unstack().sort_values(ascending=False)
    top_correlations = top_correlations[top_correlations < 1].drop_duplicates().head(3)

    for (col1, col2), corr_value in top_correlations.items():
        st.write(f"### Wykres Distplot dla: **{col1}** i **{col2}**")
        st.write(f"Współczynnik korelacji: {corr_value:.2f}")

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(df[col1], kde=True, color="blue", label=col1, ax=ax, alpha=0.5)
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(df[col2], kde=True, color="orange", label=col2, ax=ax, alpha=0.5)
        ax.legend()
        ax.set_title(f"{col1} vs {col2}")
        ax.set_xlabel("Wartości")
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
        if pca_data is not None:
            st.write("### Wizualizacja PCA i KMeans")
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(
                pca_data[:, 0], pca_data[:, 1], c=clusters, cmap='viridis', alpha=0.7
            )
            plt.colorbar(scatter, label="Klaster")
            ax.set_title("PCA i KMeans - Przypisanie do klastrów")
            ax.set_xlabel("PCA1")
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

            cluster_description, detailed_description = cluster_descriptions.get(user_cluster, ("Nieznana grupa", "Brak dodatkowych informacji."))
            st.write(f"Opis: {cluster_description}")
            st.write(detailed_description)
    else:
        st.warning("Najpierw wczytaj dane w zakładce 'Wczytaj dane'.")

def main():
    
    with st.sidebar:
        st.markdown(
            """
            <style>
            .fixed-menu {
                position: fixed;
                top: 0;
                width: 100%;
                z-index: 1000;
                background-color: white;
                box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        menu = option_menu(
            menu_title="Menu",  
            options=["Wczytaj dane", "Wykresy", "Dane predykcyjne"],  
            icons=["cloud-upload", "bar-chart-line", "robot"], 
            menu_icon="cast", 
            default_index=0,
            orientation="vertical",
        )

    if "data" not in st.session_state:
        st.session_state.data = None

    if menu == "Wczytaj dane":
        st.title("Przeanalizujmy razem Twoje dane!")
        st.divider()
        st.header("Poniżej wczytaj swój :blue[plik] i zaczynamy :sunglasses:")
        st.divider()
        data = load_data()
        if data is not None:
            display_data(data)
            filter_data_by_user_id(data)

    elif menu == "Wykresy":
        if st.session_state.data is not None:
            plot_correlated_charts(st.session_state.data)
        else:
            st.warning("Najpierw wczytaj dane w zakładce 'Wczytaj dane'.")

    elif menu == "Dane predykcyjne":
        predictive_data_section()

if __name__ == "__main__":
    main()