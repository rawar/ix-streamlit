import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from stqdm import stqdm
import lifetimes
from lifetimes.plotting import plot_period_transactions
from lifetimes.utils import calibration_and_holdout_data
from PIL import Image
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from plotly.graph_objs import XAxis, YAxis, ZAxis, Scene


@st.cache(allow_output_mutation=True)
def get_data(uploaded_file):
    df = pd.read_csv(uploaded_file, encoding='unicode_escape')
    return df


def main():
    st.set_page_config(page_title="iX Customer Analyser", layout='wide')

    logo = Image.open('app/logo_ix_web.jpg')
    st.sidebar.image(logo)
    st.sidebar.title("Customer Analyser")
    st.sidebar.markdown("""Diese Anwendung ermöglicht die interaktive Datenanalyse von [Online Retail](https://www.kaggle.com/vijayuv/onlineretail) Daten.
  Mit Hilfe der _Checkboxen_ lassen sich die einzelnen Analysen ein- und wieder ausschalten. Die Erzeugung von Graphen kann dabei etwas Zeit in Anspruch nehmen.
  """)

    # set app layout
    col1, col2, col3 = st.beta_columns([3, 1, 1])

    # initialize dataframe
    df = pd.DataFrame()
    data_loaded = False

    col1.markdown("# Datenexploration")
    status_slot = col1.empty()
    col1.markdown("""Für die geladenen Daten stehen folgende Analysen zur Verfügung:
  """)

    st.sidebar.markdown("""## Daten laden
  Daten in Form einer CSV-Datei können entweder per _drag and drop_ hier abgelegt oder über die Schaltfläche _Browse files_ geladen werden.
  """)
    uploaded_file = st.sidebar.file_uploader("", type=["csv"])
    if uploaded_file is not None:
        df = get_data(uploaded_file)
        status_slot.text("Lade Daten...")
        # Data cleaning
        df.dropna(axis=0, subset=["CustomerID"], inplace=True)

        # Convert InvoiceDate
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%m/%d/%Y %H:%M')
        df.loc[:, "Month"] = df.InvoiceDate.dt.month
        df.loc[:, "Time"] = df.InvoiceDate.dt.time
        df.loc[:, "Year"] = df.InvoiceDate.dt.year
        df.loc[:, "Day"] = df.InvoiceDate.dt.day
        df.loc[:, "Quarter"] = df.InvoiceDate.dt.quarter
        df.loc[:, "Day of Week"] = df.InvoiceDate.dt.dayofweek

        # calculate total amount
        df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

        data_loaded = True
        status_slot.markdown("<font color='green'>Daten sind geladen</font>", unsafe_allow_html=True)

    col1.markdown("## 1. Teile des Datensatzes in tabellenform Anzeigen")
    # display the dataset
    if col1.checkbox("Datensatz anzeigen"):
        if data_loaded:
            # only rows without NaN values
            # the sample file is a little bit to big
            countries = df["Country"].unique()
            country = col1.selectbox("Land auswählen", options=countries, index=0)
            col1.dataframe(df[df["Country"] == country].sample(frac=0.15))
        else:
            status_slot.markdown("<font color='red'>Daten müssen zuerst geladen werden</font>", unsafe_allow_html=True)

    col1.markdown("## 2. Geigenplot nach Preis und Land")
    if col1.checkbox("Daten untersuchen"):
        if data_loaded:
            countries = df["Country"].unique()
            fig, ax = plt.subplots(10, 4, figsize=(18, 20))
            axes_ = [axes_row for axes in ax for axes_row in axes]

            with col1:
                for i, c in enumerate(stqdm(countries, desc="Erzeuge Plots")):
                    sns.violinplot(x="UnitPrice", data=df[df["Country"] == c], ax=axes_[i], inner="point",
                                   palette="pastel")
                    axes_[i].set_title(f"Verteilung der Preise in {c}")
                    plt.tight_layout()
                col1.pyplot(fig)
        else:
            status_slot.markdown("<font color='red'>Daten müssen zuerst geladen werden</font>", unsafe_allow_html=True)

    col1.markdown("## 3. Top Produktkäufe")
    if col1.checkbox("Top Produktkäufe"):
        if data_loaded:
            top = col1.number_input('Die Top-N Produktkäufe', min_value=1, max_value=100, value=10)

            fig = plt.figure(figsize=(top, 5))
            product_hist = df.groupby("Description").sum().sort_values(by="Quantity", ascending=False).head(top)[
                "Quantity"]
            product_hist = product_hist.reset_index()
            sns.barplot(x='Quantity', y='Description', data=product_hist, label='Test', color='b', edgecolor='w',
                        orient='h')
            sns.set(style="darkgrid")
            col1.pyplot(fig)
        else:
            status_slot.markdown("<font color='red'>Daten müssen zuerst geladen werden</font>", unsafe_allow_html=True)

    col1.markdown("## 4. Top Länder nach Produktkäufe")
    if col1.checkbox("Top-10 Ländern nach Produktkäufen"):
        if data_loaded:
            product_country_hist = df.groupby("Country").sum().sort_values(by="Quantity", ascending=False).head(10)[
                "Quantity"]
            product_country_hist = product_country_hist.reset_index()
            col1.dataframe(product_country_hist)
        else:
            status_slot.markdown("<font color='red'>Daten müssen zuerst geladen werden</font>", unsafe_allow_html=True)

    col1.markdown("## 5. Top-10 Produkte innerhalb der Länder mit den meisten Produktkäufen")
    if col1.checkbox("Top-10 Produkte innerhalb der Top-10 Länder"):
        if data_loaded:
            product_country_hist = \
                df.groupby(["Country", "Description"]).sum().sort_values(by="Quantity", ascending=False)["Quantity"]
            product_country_hist = product_country_hist.reset_index()
            fig, ax = plt.subplots(5, 2, figsize=(18, 20))
            axes_ = [axes_row for axes in ax for axes_row in axes]
            top_10_countries = product_country_hist["Country"].unique()[:10]

            with col1:
                for i, c in enumerate(stqdm(top_10_countries, desc="Erzeuge Plots")):
                    top_10_products = product_country_hist[product_country_hist["Country"] == c].head(10)
                    sns.barplot(x='Quantity', y='Description', data=top_10_products, ax=axes_[i], label='Test',
                                color='b', edgecolor='w', orient='h')
                    axes_[i].set_title(f"Meist verkaufte Produkte in {c}")
                    plt.tight_layout()

                col1.pyplot(fig)
        else:
            status_slot.markdown("<font color='red'>Daten müssen zuerst geladen werden</font>", unsafe_allow_html=True)

    col1.markdown("## 6. Käufe nach Zeiträumen analysieren")
    if col1.checkbox("Analyse durchführen"):
        if data_loaded:

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

            df["Day of Week"] = df["Day of Week"].map(dayofweek_mapping)

            fig = plt.figure(figsize=(16, 12))
            plt.subplot(3, 2, 1)
            sns.lineplot(x="Month", y="Quantity", data=df.groupby("Month").sum("Quantity"), marker="o")
            plt.title("Käufe pro Monat")

            plt.subplot(3, 2, 2)
            df.groupby("Year").sum()["Quantity"].plot(kind="bar")
            plt.title("Käufe pro Jahr")

            plt.subplot(3, 2, 3)
            df.groupby("Quarter").sum()["Quantity"].plot(kind="bar")
            plt.title("Käufe pro Quartal")

            plt.subplot(3, 2, 4)
            sns.lineplot(x="Day", y="Quantity", data=df.groupby("Day").sum("Quantity"), marker="o", )
            plt.title("Käufe pro Tag")

            plt.subplot(3, 2, 5)
            df.groupby("Day of Week").sum()["Quantity"].plot(kind="bar")
            plt.title("Käufe pro Wochentag")
            plt.tight_layout()
            col1.pyplot(fig)
        else:
            status_slot.markdown("<font color='red'>Daten müssen zuerst geladen werden</font>", unsafe_allow_html=True)

    col1.markdown("## 7. Kundengruppen analysieren")
    if col1.checkbox("RFM Analyse"):
        col1.markdown("""__RFM__ steht für __Recency__ (Aktualität), __Frequency__ und __Monetary__ Score derKunden
                      Die RFM-Analyse kann helfen, die wichtigsten Kundengruppen und unwichtigsten Kundengruppen zu verstehen.
*  __Recency__: Gibt die Kaufaktivität des  Kunden an.
* __Frequency__: Gibt die Häufigkeit bzw. die Anzahö der Produktbestellungen an
* __Monetary__: Gibt die Gesamtausgaben eines Kunden an
                  """)

        if data_loaded:
            rfm_summary = lifetimes.utils.summary_data_from_transaction_data(df, "CustomerID", "InvoiceDate",
                                                                             "TotalAmount")
            rfm_summary.reset_index(inplace=True)
            fig = plt.figure(figsize=(14, 8))
            plt.subplot(221)
            sns.distplot(rfm_summary["frequency"])
            plt.title("Frequency Distribution")

            plt.subplot(222)
            sns.distplot(rfm_summary["recency"])
            plt.title("Recency Distribution")

            plt.subplot(223)
            sns.distplot(rfm_summary["T"])
            plt.title("T Distribution")

            plt.subplot(224)
            sns.distplot(rfm_summary["monetary_value"])
            plt.title("Monetary Value Distribution")
            plt.tight_layout()
            col1.pyplot(fig)
        else:
            status_slot.markdown("<font color='red'>Daten müssen zuerst geladen werden</font>", unsafe_allow_html=True)

    col1.markdown("## 8. Clustering")
    if col1.checkbox("Erstelle Cluster"):
        if data_loaded:
            clusters = col1.number_input("Anzahl Cluster", min_value=2, max_value=10, value=5)
            iterations = col1.number_input("Anzahl Iterationen", min_value=10, max_value=1000, value=50)
            kmeans_model = KMeans(n_clusters=clusters, max_iter=iterations)
            rfm_summary = lifetimes.utils.summary_data_from_transaction_data(df, "CustomerID", "InvoiceDate",
                                                                             "TotalAmount")
            rfm_summary.reset_index(inplace=True)
            kmeans_model.fit(rfm_summary)

            rfm_summary.index = pd.RangeIndex(len(rfm_summary.index))
            rfm_cluster_df = pd.concat([rfm_summary, pd.Series(kmeans_model.labels_)], axis=1)
            rfm_cluster_df.columns = ['CustomerID', 'Frequency', 'Recency', 'T', 'Amount', 'ClusterID']
            col1.dataframe(rfm_cluster_df)


            fig = go.Figure(data=[go.Surface()])

            for c in list(rfm_cluster_df['ClusterID'].unique()):
                fig.add_trace(go.Scatter3d(
                    x=rfm_cluster_df[rfm_cluster_df['ClusterID'] == c]['Recency'],
                    y=rfm_cluster_df[rfm_cluster_df['ClusterID'] == c]['Amount'],
                    z=rfm_cluster_df[rfm_cluster_df['ClusterID'] == c]['Frequency'],
                    #mode='markers', marker_size=8, marker_line_width=1,
                    mode='markers',
                    marker=dict(
                        size=8,
                        line=dict(
                            width=1
                        )
                    ),
                    opacity=0.7,
                    name='ClusterID:' + str(c),
                    hovertemplate="Recency=%{x}, Amount=%{y}, Frequency=%{z}"
                ))

            fig.update_layout(title='RFM Cluster', autosize=False,
                              width=800, height=800,
                              scene=Scene(
                                  xaxis=XAxis(title='Recency'),
                                  yaxis=YAxis(title='Amount'),
                                  zaxis=ZAxis(title='Frequency')
                              ),
                              margin=dict(l=40, r=40, b=40, t=40))
            col1.plotly_chart(fig)

        else:
            status_slot.markdown("<font color='red'>Daten müssen zuerst geladen werden</font>", unsafe_allow_html=True)

    col1.markdown("## 9. Beta Geo Model")
    if col1.checkbox("Beta Geo Model"):
        if data_loaded:
            predicted_days = col1.number_input("Anzahl Tage für Vorhersage", min_value=1, max_value=30, value=30)
            penalizer_coef = col1.number_input("Penalizer Koeffizient", min_value=0.0, max_value=1.0, value=0.0)

            rfm_summary = lifetimes.utils.summary_data_from_transaction_data(df, "CustomerID", "InvoiceDate",
                                                                             "TotalAmount")
            rfm_summary.reset_index(inplace=True)

            col1.dataframe(rfm_summary)

            bgf = lifetimes.BetaGeoFitter(penalizer_coef=penalizer_coef)
            bgf.fit(rfm_summary["frequency"], rfm_summary["recency"], rfm_summary["T"])

            rfm_summary["predicted_purchases"] = bgf.conditional_expected_number_of_purchases_up_to_time(predicted_days,
                                                                                                         rfm_summary[
                                                                                                             "frequency"],
                                                                                                         rfm_summary[
                                                                                                             "recency"],
                                                                                                         rfm_summary[
                                                                                                             "T"])

            rfm_summary_cal_holdout = calibration_and_holdout_data(
                df, "CustomerID", "InvoiceDate",
                calibration_period_end='2011-06-08',
                observation_period_end='2011-12-09'
            )
            rfm_summary_cal_holdout = rfm_summary_cal_holdout.reset_index()

            bgf.fit(
                rfm_summary_cal_holdout["frequency_cal"],
                rfm_summary_cal_holdout["recency_cal"],
                rfm_summary_cal_holdout["T_cal"]
            )

            col1.dataframe(rfm_summary_cal_holdout)

        else:
            status_slot.markdown("<font color='red'>Daten müssen zuerst geladen werden</font>", unsafe_allow_html=True)


if __name__ == '__main__':
    main()
