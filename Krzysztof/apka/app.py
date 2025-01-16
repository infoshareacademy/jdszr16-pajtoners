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
        
        weight = filtered_df['weight'].iloc[0]
        height_m = filtered_df['height'].iloc[0] / 100
        bmi = weight / (height_m ** 2)
        
        max_heart = filtered_df['Heart'].max()
        
        summary_data = (
            f"Wiek: {filtered_df['age'].iloc[0]}\n"
            f"Płeć: {'mężczyzna' if filtered_df['gender'].iloc[0] == 1 else 'kobieta'}\n"
            f"Wzrost (cm): {filtered_df['height'].iloc[0]}\n"
            f"Waga (kg): {filtered_df['weight'].iloc[0]}\n"
            f"Kalkulacja BMI: {bmi:.2f}\n"
            f"Suma kroków: {filtered_df['Steps'].sum():.0f}\n"
            f"Średnie tętno: {filtered_df['Heart'].mean():.2f}\n"
            f"Maksymalne tętno: {max_heart:.2f}\n"
            f"Suma kalorii: {filtered_df['Calories'].sum():.2f}\n"
            f"Suma dystansu (km): {filtered_df['Distance'].sum():.2f}"
        )
        st.text(summary_data)
        
        return filtered_df
    else:
        st.warning(f"Brak danych dla user_id: {st.session_state.user_id}")
        st.session_state.filtered_data = None
        return None

def plot_charts(df):
    """Function to plot charts"""
    st.header("Analiza wykresów")
    st.divider()
    st.markdown("W tej sekcji prezentujemy wykresy najważniejszych korelacji między zmiennymi w zestawie danych.")
    st.divider()

    filtered_data = st.session_state.get('filtered_data')
    
    if filtered_data is not None and not filtered_data.empty:
        
        st.subheader("Proporcje aktywności użytkownika")
        if 'activity_trimmed' in filtered_data.columns:
            activity_counts = filtered_data['activity_trimmed'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(
                activity_counts,
                labels=activity_counts.index,
                autopct='%1.1f%%',
                startangle=90,
                colors=sns.color_palette("pastel")
            )
            ax.set_title("Proporcje czasu spędzonego na różnych aktywnościach")
            st.pyplot(fig)
        else:
            st.warning("Brak wymaganej kolumny 'activity_trimmed' w danych.")
        
        st.divider()
        st.subheader("Wykres tętna dla każdej aktywności")
        if 'activity_trimmed' in filtered_data.columns and 'Heart' in filtered_data.columns:
            mean_heart_by_activity = filtered_data.groupby('activity_trimmed')['Heart'].mean().reset_index()
            max_heart_by_activity = filtered_data.groupby('activity_trimmed')['Heart'].max().reset_index()
            min_heart_by_activity = filtered_data.groupby('activity_trimmed')['Heart'].min().reset_index()

            # Sort by mean heart rate descending
            mean_heart_by_activity = mean_heart_by_activity.sort_values(by='Heart', ascending=False)
            max_heart_by_activity = max_heart_by_activity.set_index('activity_trimmed').loc[mean_heart_by_activity['activity_trimmed']].reset_index()
            min_heart_by_activity = min_heart_by_activity.set_index('activity_trimmed').loc[mean_heart_by_activity['activity_trimmed']].reset_index()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=mean_heart_by_activity, x='activity_trimmed', y='Heart', ax=ax, label="Średnie tętno")
            sns.lineplot(data=max_heart_by_activity, x='activity_trimmed', y='Heart', ax=ax, color='red', marker='o', label="Maksymalne tętno")
            sns.lineplot(data=min_heart_by_activity, x='activity_trimmed', y='Heart', ax=ax, color='blue', marker='o', label="Minimalne tętno")
            ax.set_title('Średnie, maksymalne i minimalne tętno dla każdej aktywności')
            ax.set_xlabel('Rodzaj aktywności')
            ax.set_ylabel('Tętno')
            ax.legend()
            st.pyplot(fig)
        else:
            st.warning("Brak wymaganych kolumn 'activity_trimmed' lub 'Heart' w danych.")

        st.divider()
                
    else:
        st.warning("Najpierw wybierz użytkownika w zakładce 'Wczytaj dane'.")

def predictive_heart_section():
    
    st.header("Ryzyko sercowe")
    st.divider()
    st.markdown("Sekcja ta analizuje dane dotyczące Twojego tętna, wieku oraz innych cech zdrowotnych, aby ocenić ryzyko problemów sercowo-naczyniowych. Na podstawie zaawansowanego algorytmu grupowania (KMeans) użytkownicy są przypisywani do jednego z trzech klastrów, które reprezentują różne poziomy ryzyka: niskie, umiarkowane lub wysokie. Wyniki są prezentowane w formie wykresu klastrów oraz szczegółowego opisu przypisanego klastru, wraz z rekomendacjami dotyczącymi stylu życia i działań prozdrowotnych. Dzięki tej analizie możesz lepiej zrozumieć swoje ryzyko sercowe i podjąć kroki w celu jego zmniejszenia.")
    
    filtered_data = st.session_state.get('filtered_data')

    if filtered_data is not None and not filtered_data.empty:
        numeric_data = filtered_data.select_dtypes(include=[np.number]).dropna()

        if numeric_data.empty:
            st.warning("Brak wystarczających danych numerycznych do analizy PCA i KMeans.")
            return

        pca_data, clusters = apply_kmeans_heart_based(numeric_data)
        if pca_data is not None:
            st.write("### Szczegółowa analiza oceny ryzyka sercowego")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.regplot(x='Intensity', y='SDNormalizedHR', data=filtered_data, scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'}, ax=ax)
            
            scatter_handle = plt.Line2D([0], [0], color='red', label='Linia regresji')  # Uchwyty dla linii
            points_handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', alpha=0.6, label='Rytm serca')
            
            plt.title(' ')
            ax.set_xlabel('Intensywność')
            ax.set_ylabel('Wahania rytmu serca')
            plt.legend(handles=[scatter_handle, points_handle])
            plt.grid(True)
            st.pyplot(fig)
            
            user_cluster = clusters[0]
            
            cluster_descriptions = {
                0: "Niskie ryzyko sercowe: Użytkownicy w tym klastrze mają stabilne tętno spoczynkowe oraz niskie wartości znormalizowanego tętna. Wskaźniki te sugerują dobrą kondycję serca oraz niskie ryzyko wystąpienia problemów sercowo-naczyniowych. Zaleca się kontynuowanie obecnego stylu życia z naciskiem na regularną aktywność fizyczną i zbilansowaną dietę.",
                1: "Umiarkowane ryzyko sercowe: Użytkownicy w tym klastrze mają umiarkowane wartości tętna spoczynkowego oraz średnie korelacje między tętnem a aktywnością krokową. Wskazuje to na pewne wyzwania dla układu sercowo-naczyniowego, ale bez znaczących zagrożeń. Rekomenduje się włączenie umiarkowanych ćwiczeń aerobowych i kontrolowanie masy ciała. Regularne badania sercowo-naczyniowe są wskazane, aby monitorować ewentualne zmiany.",
                2: "Wysokie ryzyko sercowe: Użytkownicy mają podwyższone tętno spoczynkowe, a korelacje między aktywnością a tętnem są niewielkie lub niestabilne. Może to sugerować przeciążenie serca lub inne nieprawidłowości. Zaleca się pilną konsultację z lekarzem w celu wykonania szczegółowych badań. Ważne jest także wprowadzenie zmian w stylu życia, takich jak dieta uboga w tłuszcze nasycone i unikanie stresu."
            }

            if user_cluster in cluster_descriptions:
                st.write(f"### Model przydzielił Cię do klastra numer {user_cluster}")
                st.write(cluster_descriptions[user_cluster])
    else:
        st.warning("Najpierw wczytaj dane w zakładce 'Wczytaj dane'.")

def apply_kmeans_heart_based(df):
    """KMeans clustering based on heart-related features."""
    if df is not None:
        st.divider()
        kmeans_features = ['Heart', 'RestingHeartrate', 'NormalizedHeartrate', 'CorrelationHeartrateSteps', 'age', 'weight']

        available_features = [f for f in kmeans_features if f in df.columns]
        if len(available_features) < len(kmeans_features):
            missing = set(kmeans_features) - set(available_features)
            st.warning(f"Brakuje następujących cech: {missing}")

        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(df[available_features].dropna())

        kmeans = KMeans(n_clusters=3, random_state=42, init='k-means++', max_iter=500)
        clusters = kmeans.fit_predict(standardized_data)

        df['Cluster'] = clusters

        return standardized_data, clusters
    else:
        st.warning("Najpierw wczytaj dane w zakładce 'Wczytaj dane'.")
        return None, None

def activity_evaluation_section():
    """Evaluate user activity using PCA and clustering."""
    st.header("Ocena aktywności")
    st.divider()
    st.markdown("Sekcja ta analizuje Twoje dane związane z codzienną aktywnością, takie jak liczba kroków, spalone kalorie, dystans i tętno. Wykorzystując techniki analizy danych, takie jak PCA i algorytm grupowania (KMeans), aplikacja klasyfikuje Twoje poziomy aktywności do jednego z trzech klastrów: niskiej, umiarkowanej lub wysokiej aktywności. Wyniki są prezentowane w formie wizualizacji na wykresie oraz opisów, które pomogą Ci zrozumieć Twój obecny poziom aktywności i uzyskać wskazówki dotyczące jej optymalizacji. Dzięki temu możesz podejmować świadome decyzje o poprawie swojego stylu życia.")
    st.divider()
    filtered_data = st.session_state.get('filtered_data')

    if filtered_data is not None and not filtered_data.empty:
        st.write("### Szczegółowa analiza aktywności")

        activity_features = ['Steps', 'Distance', 'Calories', 'Heart', 'RestingHeartrate']
        available_features = [f for f in activity_features if f in filtered_data.columns]

        if len(available_features) < len(activity_features):
            missing_features = set(activity_features) - set(available_features)
            st.warning(f"Brakuje następujących cech: {missing_features}")

        weighted_data = filtered_data.copy()
        if 'Steps' in weighted_data.columns:
            weighted_data['Steps'] *= 1.5
        if 'Calories' in weighted_data.columns:
            weighted_data['Calories'] *= 1.2

        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(weighted_data[available_features].dropna())

        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(standardized_data)

        kmeans = KMeans(n_clusters=3, random_state=42, init='k-means++', max_iter=500)
        clusters = kmeans.fit_predict(pca_data)
        filtered_data['Cluster'] = clusters

        melted_data = filtered_data.melt(
            id_vars=['Cluster'], 
            value_vars=['Steps', 'Heart'], 
            var_name='Feature', 
            value_name='Value'
        )

        polish_labels = {
            'Steps': 'Kroki',
            'Heart': 'Rytm serca'
        }
        melted_data['Feature'] = melted_data['Feature'].map(polish_labels)
        
        fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=melted_data, 
            x='Feature', 
            y='Value', 
            hue='Cluster', 
            ax=ax_bar,
        )
        ax_bar.set_title("Średnie wartości kroków i rytmu serca w poszczególnych klastrach")
        ax_bar.set_xlabel("Cechy aktywności")
        ax_bar.set_ylabel("Średnia wartość")
        st.pyplot(fig_bar)

        cluster_descriptions = {
            0: "Niska aktywność: Użytkownicy z minimalną liczbą kroków i spalonymi kaloriami. Zalecana zwiększona aktywność fizyczna.",
            1: "Umiarkowana aktywność: Użytkownicy o średnich wartościach aktywności. Dobra równowaga między aktywnością a regeneracją.",
            2: "Wysoka aktywność: Użytkownicy bardzo aktywni z dużą liczbą kroków i dystansem. Zalecane monitorowanie obciążenia fizycznego."
        }

        user_cluster = filtered_data['Cluster'].iloc[0]
        if user_cluster in cluster_descriptions:
            st.write(f"### Model przydzielił Cię do klastra numer {user_cluster}")
            st.write(cluster_descriptions[user_cluster])
    else:
        st.warning("Najpierw wczytaj dane w zakładce 'Wczytaj dane'.")
        
def improving_fitness():
    """
    Funkcja Streamlit do przewidywania poprawy kondycji użytkownika na podstawie modelu ML.
    """
    st.header("Przewidywanie poprawy kondycji")

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

            st.divider()
            st.write("""
            ### Wyniki przewidywania poprawy kondycji
            Tabela poniżej przedstawia szczegóły dotyczące Twojej aktywności fizycznej oraz prognozowane wskazania, czy Twój poziom kondycji się poprawił w danym okresie.
            """)
           
            result_df = pd.DataFrame({"Data": X_test.to_numpy().tolist(), "Prediction": predictions.tolist()})
            st.write(result_df)

            improvement_count = predictions.tolist().count(1)
            no_improvement_count = predictions.tolist().count(0)

            if improvement_count > no_improvement_count:
                st.success("Gratulacje! Nasz model przewiduje, że Twoja kondycja poprawiła się w większości przypadków.")
            else:
                st.warning("Model wskazuje, że w większości przypadków brak jest poprawy kondycji. Może warto rozważyć zwiększenie liczby kroków lub dystansu?")

            
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
            
def goals_and_progress():
    """Track user goals and progress."""
    st.header("Cele i postępy")
    st.divider()
    filtered_data = st.session_state.get('filtered_data')

    if filtered_data is None or filtered_data.empty:
        st.warning("Najpierw wczytaj dane i wybierz użytkownika.")
        return

    st.subheader("Ustaw swoje cele")
    steps_goal = st.number_input("Cel kroków", min_value=0, value=10000, step=1000)
    calories_goal = st.number_input("Cel kalorii", min_value=0, value=500, step=100)
    distance_goal = st.number_input("Cel dystansu (km)", min_value=0.0, value=5.0, step=0.5)

    st.subheader("Postęp w realizacji celów")
    total_steps = filtered_data['Steps'].sum()
    total_calories = filtered_data['Calories'].sum()
    total_distance = filtered_data['Distance'].sum()

    st.write(f"Twoje kroki: {total_steps:.0f} / {steps_goal} ({(total_steps / steps_goal) * 100:.2f}%)")
    st.write(f"Twoje kalorie: {total_calories:.1f} / {calories_goal} ({(total_calories / calories_goal) * 100:.2f}%)")
    st.write(f"Twój dystans: {total_distance:.2f} km / {distance_goal} km ({(total_distance / distance_goal) * 100:.2f}%)")

    fig, ax = plt.subplots()
    categories = ['Kroki', 'Kalorie', 'Dystans']
    values = [
        (total_steps / steps_goal) * 100,
        (total_calories / calories_goal) * 100,
        (total_distance / distance_goal) * 100
    ]
    ax.bar(categories, values, color=['#1b4332', '#40916c', '#74c69d'])
    ax.set_ylim(0, 150)
    plt.grid(True)
    st.pyplot(fig)

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
            options=["Wczytaj dane", "Wykresy", "Ryzyko sercowe", "Ocena aktywności", "Poprawa kondycji", "Cele i postępy", "O aplikacji"],  
            icons=["cloud-upload", "bar-chart-line", "heart", "clipboard-check", "person-arms-up", "graph-up-arrow", "gear"], 
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
        
    elif menu == "Cele i postępy":
        goals_and_progress()
    
    elif menu == "O aplikacji":
        st.header("Witaj w naszej aplikacji, stworzonej specjalnie do analizy danych pochodzących z Apple Watch! :tada:")
        st.divider()
        st.markdown("Nasza aplikacja pozwala na wczytanie pliku zawierającego dane o Twojej aktywności fizycznej, takich jak liczba kroków, pokonany dystans, tętno czy spalone kalorie. Dzięki temu możesz w prosty i przejrzysty sposób eksplorować swoje osiągnięcia i dowiedzieć się więcej o swoich nawykach ruchowych oraz zaplanować kolejne treningi! W kolejnych zakładkach znajdziesz: Wykresy: Interaktywne wizualizacje, które pomogą Ci zrozumieć, jakie aktywności wpływały na spalone kalorie i jak wyglądają zależności pomiędzy różnymi parametrami. Podsumowanie aktywności: Szczegółowe statystyki, które pozwolą Ci przeanalizować swoje wyniki i odkryć potencjalne obszary do poprawy.")
        st.markdown("Zanurz się w analizie swoich danych i odkryj, co Twoje Apple Watch ma Ci do powiedzenia! :blush:")

if __name__ == "__main__":
    main()
