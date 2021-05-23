import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from stqdm import stqdm

# https://www.heise.de/ix/
@st.cache(allow_output_mutation=True)
def get_data(uploaded_file):
    df = pd.read_csv(uploaded_file, encoding='unicode_escape')
    return df

def main():
  st.set_page_config(page_title="iX Customer Lifetime Value Predictor", layout='wide')
  st.sidebar.title("Beschreibung")
  st.sidebar.markdown("Dies ist ein __Markdown__ Text")

  col1, col2 = st.beta_columns([2,1])
  df = pd.DataFrame()

  uploaded_file = st.sidebar.file_uploader("CSV-Datei laden", type=["csv"])
  if uploaded_file is not None:
    df = get_data(uploaded_file)

    # Data cleaning
    df.dropna(axis=0, subset=["CustomerID"], inplace=True)
    # Convert InvoiceDate
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%m/%d/%Y %H:%M')


  # display the dataset
  if col1.checkbox("Datensatz anzeigen"):
    # only rows without NaN values
    # the sample file is a little bit to big
    col1.dataframe(df.sample(frac=0.15))


  if col1.checkbox("Daten untersuchen"):
    col1.write("### Violinplot nach Preis und Land")
    countries = df["Country"].unique()
    fig, ax = plt.subplots(10, 4, figsize=(18, 20))
    axes_ = [axes_row for axes in ax for axes_row in axes]
    sns.set(style="darkgrid")

    with col1:
      for i, c in enumerate(stqdm(countries, desc="Erzeuge Plots")):
        sns.violinplot(x="UnitPrice", data=df[df["Country"] == c], ax=axes_[i], inner="point", palette="pastel")
        axes_[i].set_title(f"Verteilung der Preise in {c}")
        plt.tight_layout()
      col1.pyplot(fig)

  if col1.checkbox("Top-10 Produktkäufe"):
    fig = plt.figure(figsize=(10, 8))
    product_hist = df.groupby("Description").sum().sort_values(by="Quantity", ascending=False).head(10)["Quantity"]
    product_hist = product_hist.reset_index()
    col1.dataframe(product_hist)
    #sns.barplot(x='Quantity', y='Description', data=product_hist, label='Test', color='b', edgecolor='w', orient='h')
    #sns.set(style="darkgrid")
    #col1.pyplot(fig)

  if col1.checkbox("Top-10 Ländern nach Produktkäufen"):
    fig = plt.figure(figsize=(10, 8))
    product_country_hist = df.groupby("Country").sum().sort_values(by="Quantity", ascending=False).head(10)["Quantity"]
    product_country_hist = product_country_hist.reset_index()
    col1.dataframe(product_country_hist)
    #sns.barplot(x='Quantity', y='Description', data=product_hist, label='Test', color='b', edgecolor='w', orient='h')
    #sns.set(style="darkgrid")
    #col1.pyplot(fig)

  if col1.checkbox("Zeitreihenanalyse"):
    temp_data = df.copy()
    temp_data.loc[:, "Month"] = df.InvoiceDate.dt.month
    temp_data.loc[:, "Time"] = df.InvoiceDate.dt.time
    temp_data.loc[:, "Year"] = df.InvoiceDate.dt.year
    temp_data.loc[:, "Day"] = df.InvoiceDate.dt.day
    temp_data.loc[:, "Quarter"] = df.InvoiceDate.dt.quarter
    temp_data.loc[:, "Day of Week"] = df.InvoiceDate.dt.dayofweek

    # Mapping day of week
    dayofweek_mapping = dict(
    {
      0: "Montag",
      1: "Dienstag",
      2: "Mittwoch",
      3: "Donnerstag",
      4: "Freitag",
      5: "Samstag",
      6: "Sonntag"
    })

    temp_data["Day of Week"] = temp_data["Day of Week"].map(dayofweek_mapping)

    sns.set(style="darkgrid")
    fig = plt.figure(figsize=(16, 12))
    plt.subplot(3, 2, 1)
    sns.lineplot(x="Month", y="Quantity", data=temp_data.groupby("Month").sum("Quantity"), marker="o", color="lightseagreen")

    plt.subplot(3, 2, 2)
    temp_data.groupby("Year").sum()["Quantity"].plot(kind="bar")
    plt.title("Läufe pro Jahr")

    plt.subplot(3, 2, 3)
    temp_data.groupby("Quarter").sum()["Quantity"].plot(kind="bar", color="darkslategrey")
    plt.title("Käufe pro Quartal")

    plt.subplot(3, 2, 4)
    sns.lineplot(x="Day", y="Quantity", data=temp_data.groupby("Day").sum("Quantity"), marker="o", )
    plt.axvline(7, color='r', linestyle='--')
    plt.axvline(15, color='k', linestyle="dotted")
    plt.title("Käufe pro Tag")

    plt.subplot(3, 2, 5)
    temp_data.groupby("Day of Week").sum()["Quantity"].plot(kind="bar", color="darkorange")
    plt.title("Käufe pro Wochentag")
    plt.tight_layout()
    col1.pyplot(fig)

if __name__ == '__main__':
  main()