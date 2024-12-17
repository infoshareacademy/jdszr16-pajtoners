import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    st.write(df.head())

def filter_data_by_user_id(df):
    """Function to filter data by user ID."""
    if "user_id" not in st.session_state:
        st.session_state.user_id = 0

    st.session_state.user_id = st.number_input("Wpisz user_id", min_value=0, step=1, value=st.session_state.user_id)
    if st.session_state.user_id in df['user_id'].values:
        filtered_df = df[df['user_id'] == st.session_state.user_id]
        st.subheader(f"Dane dla user_id: {st.session_state.user_id}")
        st.write(filtered_df)
        st.subheader("Podstawowe informacje o przefiltrowanych danych")
        st.write(filtered_df.describe().T)
        return filtered_df
    else:
        st.warning(f"Brak danych dla user_id: {st.session_state.user_id}")
        return df

def plot_correlated_charts(df):
    st.header("Analiza wykresów")
    st.write("W tej sekcji prezentujemy wykresy najważniejszych korelacji między zmiennymi w zestawie danych.")
    """Function to generate distplots for top correlated columns as separate images."""
    st.subheader("Najlepiej skorelowane kolumny - Wizualizacja")
    
    # Calculate correlation and select top pairs
    correlation_matrix = df.corr(numeric_only=True)
    top_correlations = correlation_matrix.abs().unstack().sort_values(ascending=False)
    top_correlations = top_correlations[top_correlations < 1].drop_duplicates().head(3)
    
    # Generate individual distplots
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

def predictive_data_section():
    """Placeholder for predictive data section."""
    st.subheader("Dane predykcyjne")
    st.write("Sekcja do danych predykcyjnych. Dodaj funkcjonalność według potrzeb.")

def main():
    """Main function"""
    
    
    # Sidebar menu
    menu = st.sidebar.radio("Menu", ["Wczytaj dane", "Wykresy", "Dane predykcyjne"])
    
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
