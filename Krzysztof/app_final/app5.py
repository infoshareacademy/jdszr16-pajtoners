from streamlit_option_menu import option_menu
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import streamlit as st

try:
    with open("kmeans_new.pkl", "rb") as f:
        kmeans = pickle.load(f)
    with open("scaler_new.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("pca_new.pkl", "rb") as f:
        pca = pickle.load(f)
    with open("kmeans2_new.pkl", "rb") as f:
        kmeans2 = pickle.load(f)
    with open("scaler2_new.pkl", "rb") as f:
        scaler2 = pickle.load(f)
    with open("pca2_new.pkl", "rb") as f:
        pca2 = pickle.load(f)
    with open("xgb_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)       
    with open("scaler_fitness.pkl", "rb") as f:
        scaler_fitness = pickle.load(f)      

except Exception as e:
    st.error(f"Błąd wczytywania modeli: {e}")

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
            f"Suma kroków: {filtered_df['Steps'].max():.0f}\n"
            f"Średnie tętno: {filtered_df['Heart'].mean():.2f}\n"
            f"Maksymalne tętno: {max_heart:.2f}\n"
            f"Suma kalorii: {filtered_df['Calories'].sum():.2f}\n"
            f"Suma dystansu (km): {filtered_df['Distance'].max():.2f}"
        )
        
        st.text(summary_data)
        
        visualize_bmi(bmi)

        return filtered_df
    else:
        st.warning(f"Brak danych dla user_id: {st.session_state.user_id}")
        st.session_state.filtered_data = None
        return None

def visualize_bmi(bmi):
    """Function to visualize BMI categories and user's BMI."""
    st.subheader("Twoje BMI w porównaniu do klasyfikacji BMI")

    categories = ["Niedowaga", "Normalna waga", "Nadwaga", "Otyłość"]
    thresholds = [18.5, 24.9, 29.9, 40]
    colors = ['#74c69d', '#40916c', '#f9c74f', '#f94144']

    fig, ax = plt.subplots(figsize=(8, 4))

    start = 0
    for category, threshold, color in zip(categories, thresholds, colors):
        width = threshold - start
        ax.barh(0, width, left=start, color=color, label=category, edgecolor='black')
        start = threshold

    ax.axvline(bmi, color='blue', linestyle='--', label=f'Twoje BMI: {bmi:.2f}')

    ax.set_yticks([])
    ax.set_xticks([0, *thresholds])
    ax.set_xlim(0, max(thresholds))
    ax.set_title("Klasyfikacja BMI")
    ax.legend(loc="upper right")

    st.pyplot(fig)

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
            ax.bar(activity_counts.index, activity_counts, color=sns.color_palette("pastel"))
            ax.set_title("Proporcje czasu spędzonego na różnych aktywnościach")

            ax.set_xlabel("Rodzaj aktywności")
            ax.set_ylabel("Liczba wystąpień")
            ax.set_xticklabels(activity_counts.index, rotation=45, ha='right')
            st.pyplot(fig) 
        else:
            st.warning("Brak wymaganej kolumny 'activity_trimmed' w danych.")
        
        st.divider()
        st.subheader("Wykres tętna dla każdej aktywności")
        if 'activity_trimmed' in filtered_data.columns and 'Heart' in filtered_data.columns:
            mean_heart_by_activity = filtered_data.groupby('activity_trimmed')['Heart'].mean().reset_index()
            max_heart_by_activity = filtered_data.groupby('activity_trimmed')['Heart'].max().reset_index()
            min_heart_by_activity = filtered_data.groupby('activity_trimmed')['Heart'].min().reset_index()

            mean_heart_by_activity = mean_heart_by_activity.sort_values(by='Heart', ascending=False)
            max_heart_by_activity = max_heart_by_activity.set_index('activity_trimmed').loc[mean_heart_by_activity['activity_trimmed']].reset_index()
            min_heart_by_activity = min_heart_by_activity.set_index('activity_trimmed').loc[mean_heart_by_activity['activity_trimmed']].reset_index()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=mean_heart_by_activity, x='activity_trimmed', y='Heart', ax=ax)
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
    """Function to predict heart risk"""
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
                0: "**Jesteś w grupie niskiego ryzyka sercowego:** Użytkownicy w tej grupie mają stabilne tętno spoczynkowe oraz niskie wartości znormalizowanego tętna. Wskaźniki te sugerują dobrą kondycję serca oraz niskie ryzyko wystąpienia problemów sercowo-naczyniowych. Zaleca się kontynuowanie obecnego stylu życia z naciskiem na regularną aktywność fizyczną i zbilansowaną dietę.",
                1: "**Jesteś grupie umiarkowanego ryzyka sercowego:** Użytkownicy w tej grupie mają umiarkowane wartości tętna spoczynkowego oraz średnie korelacje między tętnem a aktywnością krokową. Wskazuje to na pewne wyzwania dla układu sercowo-naczyniowego, ale bez znaczących zagrożeń. Rekomenduje się włączenie umiarkowanych ćwiczeń aerobowych i kontrolowanie masy ciała. Regularne badania sercowo-naczyniowe są wskazane, aby monitorować ewentualne zmiany.",
                2: "**Jesteś w grupie wysokiego ryzyka sercowego:** Użytkownicy w tej grupie mają podwyższone tętno spoczynkowe, a korelacje między aktywnością a tętnem są niewielkie lub niestabilne. Może to sugerować przeciążenie serca lub inne nieprawidłowości. Zaleca się pilną konsultację z lekarzem w celu wykonania szczegółowych badań. Ważne jest także wprowadzenie zmian w stylu życia, takich jak dieta uboga w tłuszcze nasycone i unikanie stresu."
            }

            if user_cluster in cluster_descriptions:
                st.write(cluster_descriptions[user_cluster])
    else:
        st.warning("Najpierw wczytaj dane w zakładce 'Wczytaj dane'.")

def apply_kmeans_heart_based(df):
    """KMeans clustering based on heart-related features."""
    if df is not None:
        st.divider()
        kmeans_features = ['Heart', 'RestingHeartrate', 'NormalizedHeartrate', 'CorrelationHeartrateSteps', 'age', 'weight']
        standardized_data = scaler.transform(df[kmeans_features].dropna())
        pca_data = pca.transform(standardized_data)
        clusters = kmeans.predict(pca_data)
        df['Cluster'] = clusters
        
        return pca_data, clusters
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
        available_features = list(scaler2.feature_names_in_)

        if len(available_features) < len(activity_features):
            missing_features = set(activity_features) - set(available_features)
            st.warning(f"Brakuje następujących cech: {missing_features}")

        weighted_data = filtered_data.copy()
        if 'Steps' in weighted_data.columns:
            weighted_data['Steps'] *= 1.5
        if 'Calories' in weighted_data.columns:
            weighted_data['Calories'] *= 1.2

        standardized_data = scaler2.transform(weighted_data[available_features].dropna())
        pca_data = pca2.transform(standardized_data)
        clusters = kmeans2.predict(pca_data)
        filtered_data['Cluster'] = clusters

        cluster_means = filtered_data.groupby('Cluster')[activity_features].mean()
        cluster_means = cluster_means.T  
        
        fig, ax = plt.subplots(figsize=(12, 6))
        cluster_means.plot(kind='bar', ax=ax)
        ax.set_title("Porównanie średnich wartości cech aktywności między klastrami")
        ax.set_xlabel("Cechy aktywności")
        ax.set_ylabel("Średnia wartość")
        ax.legend(title="Grupy")
        st.pyplot(fig)

        cluster_descriptions = {
            0: "**Grupa niskiej aktywności:** Niska liczba kroków, mała ilość spalonych kalorii i krótki dystans. Zaleca się zwiększenie poziomu aktywności poprzez regularne spacery, lekkie ćwiczenia aerobowe lub wprowadzenie codziennych nawyków ruchowych, np. chodzenie po schodach zamiast windy.",
            1: "**Grupa umiarkowanej aktywności:** Średni poziom aktywności, stabilna równowaga między ruchem a odpoczynkiem. Dobrze jest kontynuować obecną rutynę i stopniowo zwiększać intensywność ćwiczeń, np. poprzez dodanie treningów siłowych lub interwałowych.",
            2: "**Grupa wysokiej aktywności:** Wysoka liczba kroków, długi dystans i duże spalanie kalorii. Należy zwrócić uwagę na odpowiednią regenerację organizmu, unikanie przetrenowania oraz dbanie o nawodnienie i zbilansowaną dietę. Warto także monitorować oznaki przeciążenia, takie jak przewlekłe zmęczenie lub bóle mięśniowe."
        }

        user_cluster = filtered_data['Cluster'].iloc[0]
        if user_cluster in cluster_descriptions:
            st.write(cluster_descriptions[user_cluster])
    else:
        st.warning("Najpierw wczytaj dane w zakładce 'Wczytaj dane'.")

def improving_fitness():
    """Function to predtict improving fitness by person"""
    st.header("Przewidywanie poprawy kondycji")

    if st.button("Przewiduj wyniki"):
        try:
            filtered_data = st.session_state.get('filtered_data')

            if filtered_data is None or filtered_data.empty:
                st.error("Dane użytkownika nie zostały wczytane. Wróć do zakładki 'Wczytaj dane' i wybierz użytkownika.")
                return

            selected_columns = ['Steps', 'Distance', 'Calories', 'Heart', 'RestingHeartrate', 'age', 'weight']
            filtered_data = filtered_data[selected_columns]
            filtered_data.dropna(inplace=True)

            X_user = scaler_fitness.transform(filtered_data)
            predictions = xgb_model.predict(X_user)

            st.divider()
            st.write("### Wyniki przewidywania poprawy kondycji")
            result_df = pd.DataFrame({"Data": filtered_data.to_numpy().tolist(), "Prediction": predictions.tolist()})
            st.write(result_df)

            improvement_count = predictions.tolist().count(1)
            no_improvement_count = predictions.tolist().count(0)

            if improvement_count > no_improvement_count:
                st.success("Gratulacje! Biorąc pod uwagę takie cechy jak: ilość kroków, przemierzony dystatns, rytm bicia serca oraz wagę, wzrost i wiek, nasz model przewiduje, że Twoja kondycja poprawiła się w większości przypadków.")
            else:
                st.warning("Niestety, ale biorąc pod uwagę takie cechy jak: ilość kroków, przemierzony dystatns, rytm bicia serca oraz wagę, wzrost i wiek, nasz model wskazuje, że w większości przypadków brak jest poprawy kondycji. Może warto rozważyć zwiększenie liczby kroków lub dystansu?")

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
    total_steps = filtered_data['Steps'].max()
    total_calories = filtered_data['Calories'].sum()
    total_distance = filtered_data['Distance'].max()

    st.write(f"Twoje kroki: {total_steps:.0f} / {steps_goal} ({(total_steps / steps_goal) * 100:.2f}%)")
    st.write(f"Twoje kalorie: {total_calories:.1f} / {calories_goal} ({(total_calories / calories_goal) * 100:.2f}%)")
    st.write(f"Twój dystans: {total_distance:.2f} km / {distance_goal} km ({(total_distance / distance_goal) * 100:.2f}%)")

    actual_values = [total_steps, total_calories, total_distance]
    categories = ['Kroki', 'Kalorie', 'Dystans']

    max_steps = filtered_data['Steps'].max()
    max_calories = filtered_data['Calories'].max()
    max_distance = filtered_data['Distance'].max()
    
    actual_percentages = [
        (total_steps / max_steps) * 100 if max_steps > 0 else 0,
        (total_calories / max_calories) * 100 if max_calories > 0 else 0,
        (total_distance / max_distance) * 100 if max_distance > 0 else 0
    ]

    goal_percentages = [
        (steps_goal / max_steps) * 100 if max_steps > 0 else 0,
        (calories_goal / max_calories) * 100 if max_calories > 0 else 0,
        (distance_goal / max_distance) * 100 if max_distance > 0 else 0
    ]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    goal_percentages += goal_percentages[:1]  
    actual_percentages += actual_percentages[:1]  
    angles += angles[:1]

    ax.fill(angles, goal_percentages, color='green', alpha=0.4, label='Cele użytkownika')
    ax.plot(angles, goal_percentages, color='green', linewidth=2)
    
    ax.fill(angles, actual_percentages, color='red', alpha=0.4, label='Aktualne osiągnięcia')
    ax.plot(angles, actual_percentages, color='red', linewidth=2)
    
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title("Postęp w realizacji celów")
    ax.legend()

    st.pyplot(fig)

def what_if_section():
    """Function to predict steps and calories"""
    st.header("Sprawdź swoje możliwości")
    st.divider()
    st.markdown("W tej sekcji możesz sprawdzić, ile kroków musisz zrobić, aby osiągnąć określoną liczbę spalonych kalorii, lub ile kalorii spalisz przy zadanej liczbie kroków. Obliczenia opierają się na średnich danych dostępnych w internecie.")

    avg_calories_per_step = 0.04  
    avg_steps_per_km = 1312  

    st.write(f"### Przyjęte średnie wartości do obliczeń:")
    st.write(f"- Średnie spalanie kalorii na krok: **{avg_calories_per_step:.4f} kcal**")
    st.write(f"- Średnia liczba kroków na kilometr: **{avg_steps_per_km} kroków**")

    option = st.radio("Co chcesz obliczyć?", ["Ilość kroków na spalenie kalorii", "Spalone kalorie dla zadanych kroków"])

    if option == "Ilość kroków na spalenie kalorii":
        target_calories = st.number_input("Wpisz liczbę kalorii do spalenia:", min_value=0.0, step=10.0)
        if target_calories > 0:
            required_steps = target_calories / avg_calories_per_step
            st.success(f"Aby spalić **{target_calories:.1f} kcal**, musisz zrobić około **{required_steps:.0f} kroków**.")

    elif option == "Spalone kalorie dla zadanych kroków":
        target_steps = st.number_input("Wpisz liczbę kroków:", min_value=0, step=100)
        if target_steps > 0:
            burned_calories = target_steps * avg_calories_per_step
            st.success(f"Dla **{target_steps} kroków** spalisz około **{burned_calories:.1f} kcal**.")

    st.divider()
    st.subheader("Obliczenia związane z dystansem")
    st.markdown("Dodatkowo, na podstawie liczby kroków możemy oszacować, jaki dystans pokonasz.")

    distance_steps = st.number_input("Podaj liczbę kroków, aby oszacować dystans (w kilometrach):", min_value=0, step=100)
    if distance_steps > 0:
        estimated_distance = distance_steps / avg_steps_per_km
        st.success(f"Dla **{distance_steps} kroków** oszacowany dystans wynosi około **{estimated_distance:.2f} km**.")

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
            options=["Wczytaj dane", "Wykresy", "Ryzyko sercowe", "Ocena aktywności", "Poprawa kondycji", "Cele i postępy","Sprawdź swoje możliwości","O aplikacji"],  
            icons=["cloud-upload", "bar-chart-line", "heart", "clipboard-check", "person-arms-up", "graph-up-arrow","calculator", "gear"], 
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

    elif menu == "Sprawdź swoje możliwości":
        what_if_section()
    
    elif menu == "O aplikacji":
        st.header("Witaj w naszej aplikacji, stworzonej specjalnie do analizy danych pochodzących z Apple Watch! :tada:")
        st.divider()
        st.markdown("Nasza aplikacja umożliwia wczytanie pliku z danymi o Twojej aktywności fizycznej, takimi jak liczba kroków, pokonany dystans, tętno czy spalone kalorie. Dzięki temu możesz w prosty sposób eksplorować swoje osiągnięcia, zrozumieć swoje nawyki ruchowe i zaplanować kolejne kroki w drodze do lepszej kondycji!")
        st.markdown("W naszej aplikacji znajdziesz następujące zakładki:")
        st.markdown(" - Wykresy: Interaktywne wizualizacje, które pomogą Ci zrozumieć, jakie aktywności wpływają na spalone kalorie oraz odkryć zależności między różnymi parametrami Twojej aktywności.")
        st.markdown("- Ryzyko sercowe: Analiza danych pomagająca ocenić potencjalne ryzyko chorób sercowo-naczyniowych, oparta na Twoich wynikach i aktywności.")
        st.markdown("- Ocena aktywności: Szczegółowe statystyki, które wskażą, jak dobrze realizujesz swoje cele aktywności fizycznej oraz gdzie można wprowadzić ulepszenia.")
        st.markdown("- Poprawa kondycji: Moduł oceniający, czy Twoja kondycja poprawiła się na podstawie analizy danych historycznych, oraz wskazówki, jak dalej ją rozwijać.")
        st.markdown("- Cele i postępy: Dzięki interaktywnemu wykresowi, śledzisz swoje postępy i widzisz, w jakim stopniu udało Ci się zrealizować plan treningowy. ")
        st.markdown("- Sprawdź swoje możliwości: Chcesz wiedzieć ile kcal spalisz wykonując określoną liczbę kroków? Skorzystaj z tej zakładki, jest to idealne narzędzie do zaplanowania aktywności fizycznej dostosowanej do Twoich potrzeb! ")
        st.markdown("Zanurz się w analizie swoich danych i odkryj, co Twoje Apple Watch ma Ci do powiedzenia! :blush:")

if __name__ == "__main__":
    main()
