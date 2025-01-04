from streamlit_option_menu import option_menu
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

    # Dynamic `user_id`
    st.session_state.user_id = st.number_input("Wpisz user_id", min_value=0, step=1, value=st.session_state.user_id)
    
    if st.session_state.user_id in df['user_id'].values:
        filtered_df = df[df['user_id'] == st.session_state.user_id]
        st.session_state.filtered_data = filtered_df

        st.subheader("Podsumowanie dla wybranego użytkownika")
        
        summary_text = (
            f"Wiek: {filtered_df['age'].iloc[0]}\n"
            f"Płeć: {'mężczyzna' if filtered_df['gender'].iloc[0] == 1 else 'kobieta'}\n"
            f"Wzrost (cm): {filtered_df['height'].iloc[0]}\n"
            f"Waga (kg): {filtered_df['weight'].iloc[0]}\n"
            f"Suma kroków: {filtered_df['Steps'].sum():.0f}\n"
            f"Średnie tętno: {filtered_df['Heart'].mean():.2f}\n"
            f"Suma kalorii: {filtered_df['Calories'].sum():.2f}\n"
            f"Suma dystansu (km): {filtered_df['Distance'].sum():.2f}"
        )
        st.text_area("Szczegóły użytkownika:", value=summary_text, height=200, disabled=True)
        return filtered_df
    else:
        st.warning(f"Brak danych dla user_id: {st.session_state.user_id}")
        st.session_state.filtered_data = None
        return None

def plot_charts(df):
    """Function to plot charts"""
    st.header("Analiza wykresów")
    st.subheader("W tej sekcji prezentujemy wykresy najważniejszych korelacji między zmiennymi w zestawie danych.")

    filtered_data = st.session_state.get('filtered_data')
    
    if filtered_data is not None and not filtered_data.empty:
        st.subheader("Średnie tętno dla każdej aktywności")
        if 'activity_trimmed' in filtered_data.columns and 'Heart' in filtered_data.columns:
            mean_heart_by_activity = filtered_data.groupby('activity_trimmed')['Heart'].mean().reset_index()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=mean_heart_by_activity, x='activity_trimmed', y='Heart', ax=ax)
            ax.set_title('Średnie tętno dla każdej aktywności')
            ax.set_xlabel('Rodzaj aktywności')
            ax.set_ylabel('Średnie tętno')
            st.pyplot(fig)
        else:
            st.warning("Brak wymaganych kolumn 'activity_trimmed' lub 'Heart' w danych.")
    else:
        st.warning("Najpierw wybierz użytkownika w zakładce 'Wczytaj dane'.")

def predictive_heart_section():
    
    st.subheader("Ryzyko sercowe")
    filtered_data = st.session_state.get('filtered_data')

    if filtered_data is not None and not filtered_data.empty:
        numeric_data = filtered_data.select_dtypes(include=[np.number]).dropna()

        if numeric_data.empty:
            st.warning("Brak wystarczających danych numerycznych do analizy PCA i KMeans.")
            return

        pca_data, clusters = apply_kmeans_heart_based(numeric_data)
        if pca_data is not None:
            st.write("### Wizualizacja KMeans")
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(
                pca_data[:, 0], pca_data[:, 1], c=clusters, cmap='viridis', alpha=0.7
            )
            plt.colorbar(scatter, label="Klaster")
            ax.set_title("KMeans - Przypisanie do klastrów")
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")
            st.pyplot(fig)

            user_cluster = clusters[0]
            st.write(f"Nasz program przypisał Cię do klastra: {user_cluster}")

            cluster_descriptions = {
                0: "Niskie ryzyko sercowe: Użytkownicy w tym klastrze mają stabilne tętno spoczynkowe oraz niskie wartości znormalizowanego tętna. Wskaźniki te sugerują dobrą kondycję serca oraz niskie ryzyko wystąpienia problemów sercowo-naczyniowych. Zaleca się kontynuowanie obecnego stylu życia z naciskiem na regularną aktywność fizyczną i zbilansowaną dietę.",
                1: "Umiarkowane ryzyko sercowe: Użytkownicy w tym klastrze mają umiarkowane wartości tętna spoczynkowego oraz średnie korelacje między tętnem a aktywnością krokową. Wskazuje to na pewne wyzwania dla układu sercowo-naczyniowego, ale bez znaczących zagrożeń. Rekomenduje się włączenie umiarkowanych ćwiczeń aerobowych i kontrolowanie masy ciała. Regularne badania sercowo-naczyniowe są wskazane, aby monitorować ewentualne zmiany.",
                2: "Wysokie ryzyko sercowe: Użytkownicy mają podwyższone tętno spoczynkowe, a korelacje między aktywnością a tętnem są niewielkie lub niestabilne. Może to sugerować przeciążenie serca lub inne nieprawidłowości. Zaleca się pilną konsultację z lekarzem w celu wykonania szczegółowych badań. Ważne jest także wprowadzenie zmian w stylu życia, takich jak dieta uboga w tłuszcze nasycone i unikanie stresu."
            }

            if user_cluster in cluster_descriptions:
                st.write(f"### Opis klastra {user_cluster}")
                st.write(cluster_descriptions[user_cluster])
    else:
        st.warning("Najpierw wczytaj dane w zakładce 'Wczytaj dane'.")

def apply_kmeans_heart_based(df):
    """KMeans clustering based on heart-related features."""
    if df is not None:
        st.write("Przeprowadzamy analizę KMeans...")
        kmeans_features = ['Heart', 'RestingHeartrate', 'NormalizedHeartrate', 'CorrelationHeartrateSteps', 'age', 'weight']

        # Check for missing features
        available_features = [f for f in kmeans_features if f in df.columns]
        if len(available_features) < len(kmeans_features):
            missing = set(kmeans_features) - set(available_features)
            st.warning(f"Brakuje następujących cech: {missing}")

        # Standardize the data
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(df[available_features].dropna())

        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=3, random_state=42, init='k-means++', max_iter=500)
        clusters = kmeans.fit_predict(standardized_data)

        # Add clusters to the original dataframe
        df['Cluster'] = clusters

        return standardized_data, clusters
    else:
        st.warning("Najpierw wczytaj dane w zakładce 'Wczytaj dane'.")
        return None, None

def activity_evaluation_section():
    """Evaluate user activity using PCA and clustering."""
    st.subheader("Ocena aktywności")
    filtered_data = st.session_state.get('filtered_data')

    if filtered_data is not None and not filtered_data.empty:
        st.write("### Szczegółowa analiza aktywności")

        # Select features for PCA and clustering
        activity_features = ['Steps', 'Distance', 'Calories', 'Heart', 'RestingHeartrate']
        available_features = [f for f in activity_features if f in filtered_data.columns]

        if len(available_features) < len(activity_features):
            missing_features = set(activity_features) - set(available_features)
            st.warning(f"Brakuje następujących cech: {missing_features}")

        # Increase weights for critical features
        weighted_data = filtered_data.copy()
        if 'Steps' in weighted_data.columns:
            weighted_data['Steps'] *= 1.5
        if 'Calories' in weighted_data.columns:
            weighted_data['Calories'] *= 1.2

        # Standardize the data
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(weighted_data[available_features].dropna())

        # Apply PCA
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(standardized_data)

        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=3, random_state=42, init='k-means++', max_iter=500)
        clusters = kmeans.fit_predict(pca_data)
        filtered_data['Cluster'] = clusters

        # Visualize PCA results
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            pca_data[:, 0], pca_data[:, 1], c=clusters, cmap='viridis', alpha=0.7
        )
        plt.colorbar(scatter, label="Klaster")
        ax.set_title("PCA i KMeans - Aktywność użytkowników")
        ax.set_xlabel("PCA1")
        ax.set_ylabel("PCA2")
        st.pyplot(fig)

        # Add descriptions for clusters
        cluster_descriptions = {
            0: "Niska aktywność: Użytkownicy z minimalną liczbą kroków i spalonymi kaloriami. Zalecana zwiększona aktywność fizyczna.",
            1: "Umiarkowana aktywność: Użytkownicy o średnich wartościach aktywności. Dobra równowaga między aktywnością a regeneracją.",
            2: "Wysoka aktywność: Użytkownicy bardzo aktywni z dużą liczbą kroków i dystansem. Zalecane monitorowanie obciążenia fizycznego."
        }

        # Display cluster description for the current user
        user_cluster = filtered_data['Cluster'].iloc[0]
        if user_cluster in cluster_descriptions:
            st.write(f"### Opis klastra {user_cluster}")
            st.write(cluster_descriptions[user_cluster])
    else:
        st.warning("Najpierw wczytaj dane w zakładce 'Wczytaj dane'.")
        
def improving_fitness():
    """
    Funkcja Streamlit do przewidywania poprawy kondycji użytkownika na podstawie modelu ML.
    """
    st.subheader("Przewidywanie poprawy kondycji")

    # Pobieranie danych z session_state
    try:
        filtered_data = st.session_state.get('filtered_data')

        if filtered_data is None or filtered_data.empty:
            st.error("Dane użytkownika nie zostały wczytane. Wróć do zakładki 'Wczytaj dane' i wybierz użytkownika.")
            return

        selected_columns = ['Steps', 'Distance', 'Calories', 'Heart', 'RestingHeartrate', 'age', 'weight']
        filtered_data = filtered_data[selected_columns]

        filtered_data['Improvement'] = (filtered_data['Steps'].diff().fillna(0) > 0).astype(int)
        filtered_data.dropna(inplace=True)

        X = filtered_data.drop(columns=['Improvement'])
        y = filtered_data['Improvement']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)

    except Exception as e:
        st.error(f"Wystąpił problem z przygotowaniem modelu: {e}")
        return

    if st.button("Przewiduj poprawę kondycji"):
        try:
            user_data_scaled = scaler.transform(X_test)
            predictions = model.predict(user_data_scaled)

            # Wprowadzenie
            st.write("""
            ### Wyniki przewidywania poprawy kondycji
            Tabela poniżej przedstawia szczegóły dotyczące Twojej aktywności fizycznej oraz prognozowane wskazania, czy Twój poziom kondycji się poprawił w danym okresie.
            """)
            
            # Tabela z wynikami
            result_df = pd.DataFrame({"Data": X_test.to_numpy().tolist(), "Prediction": predictions.tolist()})
            st.write(result_df)

            # Interpretacja wyników
            if 1 in predictions:
                st.success("Gratulacje! Nasz model przewiduje, że Twoja kondycja poprawiła się w oparciu o wprowadzone dane.")
            else:
                st.warning("Model nie wykrył znaczącej poprawy kondycji. Może warto rozważyć zwiększenie liczby kroków lub dystansu?")

            # Wizualizacja wyników
            improvement_count = predictions.tolist().count(1)
            no_improvement_count = predictions.tolist().count(0)
            
            fig, ax = plt.subplots()
            ax.pie(
                [improvement_count, no_improvement_count],
                labels=['Poprawa', 'Brak poprawy'],
                autopct='%1.1f%%',
                colors=['green', 'red'],
                startangle=90
            )
            ax.set_title("Przewidywana poprawa kondycji")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Wystąpił problem z przewidywaniem: {e}")

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
            options=["Wczytaj dane", "Wykresy", "Ryzyko sercowe", "Ocena aktywności", "Poprawa kondycji", "O aplikacji"],  
            icons=["cloud-upload", "bar-chart-line", "heart", "clipboard-check", "person-arms-up", "gear"], 
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
            plot_charts(st.session_state.data)
        else:
            st.warning("Najpierw wczytaj dane w zakładce 'Wczytaj dane'.")

    elif menu == "Ryzyko sercowe":
        predictive_heart_section()

    elif menu == "Ocena aktywności":
        activity_evaluation_section()
        
    elif menu == "Poprawa kondycji":
        improving_fitness()

if __name__ == "__main__":
    main()
