# ouvidoria_poc_streamlit_app.py
# Run: streamlit run ouvidoria_poc_streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk

# ----------------------------
# Global config / theming
# ----------------------------
st.set_page_config(page_title="Ouvidoria POC – Brasil", page_icon="📧", layout="wide")

PRIMARY = "#0B8F3B"  # green
ACCENT  = "#0FA958"
LIGHT   = "#F6FFF9"
DARK    = "#09331C"

def metric_card(label, value, delta=None):
    # Evita glitch de f-string no markdown
    delta_block = f"<div style='font-size:12px; color:{ACCENT}'>{delta}</div>" if delta else ""
    card_html = """
    <div style="background:{LIGHT}; padding:16px; border:1px solid #E6F4EA; border-radius:16px">
      <div style="font-size:13px; color:{DARK}; opacity:0.8">{label}</div>
      <div style="font-size:28px; font-weight:700; color:{PRIMARY}">{value}</div>
      {delta_block}
    </div>
    """.format(LIGHT=LIGHT, DARK=DARK, PRIMARY=PRIMARY, label=label, value=value, delta_block=delta_block)
    st.markdown(card_html, unsafe_allow_html=True)

# ----------------------------
# Branding (sidebar logo)
# ----------------------------
def sidebar_logo():
    """
    Coloca a logo acima dos filtros.
    Use arquivo local 'logo.png' (mesma pasta) ou defina st.secrets['logo_src'] com URL/caminho.
    """
    logo_src = st.secrets.get("logo_src", "logo.png")
    try:
        if isinstance(logo_src, str) and (logo_src.startswith("http://") or logo_src.startswith("https://")):
            st.sidebar.image(logo_src, use_container_width=True)
        else:
            from pathlib import Path as _Path
            if _Path(logo_src).exists():
                st.sidebar.image(str(logo_src), use_container_width=True)
    except Exception:
        pass
    st.sidebar.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

# ----------------------------
# Constantes geográficas
# ----------------------------
UF_TO_REGION = {
    # Norte
    "AC":"Norte","AP":"Norte","AM":"Norte","PA":"Norte","RO":"Norte","RR":"Norte","TO":"Norte",
    # Nordeste
    "AL":"Nordeste","BA":"Nordeste","CE":"Nordeste","MA":"Nordeste","PB":"Nordeste","PE":"Nordeste",
    "PI":"Nordeste","RN":"Nordeste","SE":"Nordeste",
    # Centro-Oeste
    "DF":"Centro-Oeste","GO":"Centro-Oeste","MT":"Centro-Oeste","MS":"Centro-Oeste",
    # Sudeste
    "ES":"Sudeste","MG":"Sudeste","RJ":"Sudeste","SP":"Sudeste",
    # Sul
    "PR":"Sul","RS":"Sul","SC":"Sul"
}

# ----------------------------
# Synthetic data generator
# ----------------------------
@st.cache_data(show_spinner=False)
def generate_data(n=20000, seed=42):
    np.random.seed(seed)
    # Principais cidades (lat/lon aproximados) com peso amostral
    cities = [
        ("São Paulo","SP",-23.5505,-46.6333,12.33),
        ("Rio de Janeiro","RJ",-22.9068,-43.1729,6.72),
        ("Belo Horizonte","MG",-19.9167,-43.9345,2.5),
        ("Brasília","DF",-15.7939,-47.8828,3.0),
        ("Salvador","BA",-12.9777,-38.5016,2.9),
        ("Fortaleza","CE",-3.7319,-38.5267,2.7),
        ("Curitiba","PR",-25.4284,-49.2733,1.9),
        ("Porto Alegre","RS",-30.0346,-51.2177,1.5),
        ("Recife","PE",-8.0476,-34.8770,2.1),
        ("Manaus","AM",-3.1190,-60.0217,1.4),
        ("Belém","PA",-1.4558,-48.4902,1.2),
        ("Goiânia","GO",-16.6869,-49.2648,1.1),
        ("Campinas","SP",-22.9056,-47.0608,1.1),
        ("São Luís","MA",-2.5387,-44.2825,0.9),
        ("Natal","RN",-5.7945,-35.2110,0.9),
        ("João Pessoa","PB",-7.1153,-34.8610,0.8),
        ("Maceió","AL",-9.6498,-35.7089,0.8),
        ("Florianópolis","SC",-27.5954,-48.5480,0.8),
        ("Campo Grande","MS",-20.4697,-54.6201,0.7),
        ("Cuiabá","MT",-15.6010,-56.0974,0.7),
        ("Vitória","ES",-20.3155,-40.3128,0.8),
        ("Teresina","PI",-5.0919,-42.8034,0.6),
        ("Aracaju","SE",-10.9472,-37.0731,0.6),
        ("Porto Velho","RO",-8.7608,-63.9004,0.5),
        ("Rio Branco","AC",-9.9747,-67.8093,0.4),
        ("Macapá","AP",0.0349,-51.0694,0.4),
        ("Palmas","TO",-10.1844,-48.3336,0.5)
    ]
    cities_df = pd.DataFrame(cities, columns=["city","state","lat","lon","weight"])
    cities_df["weight"] = cities_df["weight"] / cities_df["weight"].sum()

    # Amostragem por peso e jitter contido (evita oceano)
    idx = np.random.choice(len(cities_df), size=n, p=cities_df["weight"].values)
    base = cities_df.iloc[idx].reset_index(drop=True)
    base["lat"] = base["lat"] + np.random.normal(0, 0.06, size=n)  # antes 0.20
    base["lon"] = base["lon"] + np.random.normal(0, 0.06, size=n)

    # Canais — E-mail dominante (canal da OUVIDORIA)
    canal_ouvidoria = np.random.choice(["E-mail","Telefone","Site"], size=n, p=[0.65, 0.30, 0.05])

    # Tópicos / Severidade
    topics = np.random.choice(
        ["Cobrança indevida","Problema técnico","Atendimento inadequado","Cancelamento",
         "Informação de produto","Prazo/SLA","Fraude/Suspeita"],
        size=n, p=[0.24,0.22,0.18,0.12,0.10,0.09,0.05]
    )
    severity = np.random.choice(["Baixa","Média","Alta","Crítica"], size=n, p=[0.30,0.45,0.20,0.05])

    # Datas (últimos 180 dias)
    end = datetime.today()
    start = end - timedelta(days=180)
    dates = pd.to_datetime(np.random.randint(int(start.timestamp()), int(end.timestamp()), size=n), unit="s")

    # FCR / SLA (diferenças por canal da Ouvidoria)
    fcr = np.random.binomial(1, p=np.where(canal_ouvidoria=="Telefone", 0.62, 0.55), size=n).astype(bool)
    sla_ok = np.random.binomial(1, p=np.where(canal_ouvidoria=="Telefone", 0.75, 0.70), size=n).astype(bool)

    # ORIGEM (Produto/Serviço)
    produtos = ["POS","ContaPJ","Pix","Cartao","LinkPgto","Ecomm_Pagarme","Antecipacao","Controle"]
    p_prod   = [0.28, 0.18, 0.16, 0.12, 0.10, 0.08, 0.05, 0.03]
    produto_origem = np.random.choice(produtos, size=n, p=p_prod)

    # MOTIVO DO CONTATO
    motivos = ["Reclamacao","Sugestao","Critica","Denuncia","Elogio"]
    p_mtv   = [0.85, 0.05, 0.04, 0.03, 0.03]
    motivo_contato = np.random.choice(motivos, size=n, p=p_mtv)

    # CANAL PRÉVIO (antes da Ouvidoria) e PROTOCOLO PRÉVIO
    canais_previos = ["WhatsApp","Telefone","Portal","App","E-mail"]
    p_prev         = [0.35, 0.30, 0.20, 0.10, 0.05]
    canal_previo   = np.random.choice(canais_previos, size=n, p=p_prev)
    tem_protocolo_previo = np.random.binomial(1, p=0.90, size=n).astype(bool)  # 90% com protocolo

    df = pd.DataFrame({
        "timestamp": dates,
        "canal": canal_ouvidoria,              # canal na Ouvidoria
        "topico": topics,
        "severidade": severity,
        "cidade": base["city"].values,
        "uf": base["state"].values,
        "lat": base["lat"].values,
        "lon": base["lon"].values,
        "fcr": fcr,
        "sla_ok": sla_ok,
        "produto_origem": produto_origem,      # NOVO
        "motivo_contato": motivo_contato,      # NOVO
        "canal_previo": canal_previo,          # NOVO
        "tem_protocolo_previo": tem_protocolo_previo  # NOVO
    })

    # Região
    df["regiao"] = df["uf"].map(UF_TO_REGION)

    # Sentimento sintético por severidade
    mu_map = {"Baixa":0.25, "Média":0.0, "Alta":-0.35, "Crítica":-0.65}
    df["sentimento"] = np.clip([np.random.normal(mu_map[s], 0.2) for s in df["severidade"]], -1, 1)

    # Estágios para Sankey (origem → canal Ouvidoria → triagem → fila → desfecho)
    df["etapa1"] = df["canal"]
    df["etapa2"] = np.where(
        df["canal"]=="Telefone",
        np.random.choice(["URA","Atendente N1"], size=n, p=[0.6,0.4]),
        np.where(df["canal"]=="E-mail","Triagem E-mail","Formulário Web")
    )
    df["etapa3"] = np.where(
        df["topico"].isin(["Problema técnico","Fraude/Suspeita"]),
        np.random.choice(["Suporte Técnico","Equipe Fraude"], size=n, p=[0.7,0.3]),
        np.random.choice(["Backoffice","Atendimento N2"], size=n, p=[0.6,0.4])
    )
    df["desfecho"] = np.where(df["fcr"], "Resolvido 1º contato",
                        np.where(df["sla_ok"], "Resolvido dentro SLA", "Escalado/Ocorrência"))
    return df

df = generate_data()

# ----------------------------
# Sidebar filters
# ----------------------------
sidebar_logo()  # << logo acima dos filtros
st.sidebar.markdown(f"<h2 style='color:{PRIMARY}'>Filtros</h2>", unsafe_allow_html=True)

min_date, max_date = df["timestamp"].min().date(), df["timestamp"].max().date()
date_range = st.sidebar.date_input("Período", (min_date, max_date), min_value=min_date, max_value=max_date)

canal_sel   = st.sidebar.multiselect("Canal (Ouvidoria)", ["E-mail","Telefone","Site"], default=["E-mail","Telefone","Site"])
origem_sel  = st.sidebar.multiselect("Origem (Produto/Serviço)", sorted(df["produto_origem"].unique()), default=[])
motivo_sel  = st.sidebar.multiselect("Motivo do Contato", ["Reclamacao","Sugestao","Critica","Denuncia","Elogio"], default=[])
previo_sel  = st.sidebar.multiselect("Canal Prévio", ["WhatsApp","Telefone","Portal","App","E-mail"], default=[])
prot_check  = st.sidebar.checkbox("Apenas com Protocolo Prévio", value=False)

topico_sel  = st.sidebar.multiselect("Tópicos", sorted(df["topico"].unique()), default=[])
uf_sel      = st.sidebar.multiselect("UF", sorted(df["uf"].unique()), default=[])
reg_sel     = st.sidebar.multiselect("Região", ["Norte","Nordeste","Centro-Oeste","Sudeste","Sul"], default=[])
sev_sel     = st.sidebar.multiselect("Severidade", ["Baixa","Média","Alta","Crítica"], default=[])

if isinstance(date_range, tuple):
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

mask = (
    (df["timestamp"].dt.date >= start_date) &
    (df["timestamp"].dt.date <= end_date) &
    (df["canal"].isin(canal_sel)) &
    (df["produto_origem"].isin(origem_sel) if origem_sel else True) &
    (df["motivo_contato"].isin(motivo_sel) if motivo_sel else True) &
    (df["canal_previo"].isin(previo_sel) if previo_sel else True) &
    ((df["tem_protocolo_previo"] == True) if prot_check else True) &
    (df["topico"].isin(topico_sel) if topico_sel else True) &
    (df["uf"].isin(uf_sel) if uf_sel else True) &
    (df["regiao"].isin(reg_sel) if reg_sel else True) &
    (df["severidade"].isin(sev_sel) if sev_sel else True)
)

f = df.loc[mask].copy()

# ----------------------------
# Header
# ----------------------------
left, right = st.columns([0.8,0.2])
with left:
    st.markdown(f"<h1 style='color:{PRIMARY}; margin-bottom:0'>Ouvidoria POC – Brasil</h1>", unsafe_allow_html=True)
    st.caption("Heatmap (top-down) por canal, filtros por Origem/Motivo/Canal Prévio/Protocolo e KPIs executivos.")
with right:
    st.write(""); st.write("")
    # st.markdown(f"<div style='text-align:right'><span style='background:{PRIMARY}; color:white; padding:6px 10px; border-radius:10px;'>Tema: Verde & Branco</span></div>", unsafe_allow_html=True)

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2 = st.tabs(["Mapa & KPIs", "Fluxos & Prioridades"])

with tab1:
    # KPIs
    k1,k2,k3,k4,k5 = st.columns(5)
    metric_card("Volume (filtros)", f"{len(f):,}".replace(",","."))
    metric_card("FCR (1º contato)", f"{(f['fcr'].mean()*100 if len(f)>0 else 0):.1f}%")
    metric_card("SLA cumprido", f"{(f['sla_ok'].mean()*100 if len(f)>0 else 0):.1f}%")
    metric_card("Sentimento médio", f"{(f['sentimento'].mean() if len(f)>0 else 0):.2f}")
    metric_card("Conformidade protocolo", f"{(f['tem_protocolo_previo'].mean()*100 if len(f)>0 else 0):.1f}%")

    # Alerta simples de conformidade
    if len(f) and f["tem_protocolo_previo"].mean() < 0.85:
        st.warning("Conformidade de protocolo prévio abaixo do limiar de 85% para o recorte atual.")

    st.markdown("---")

    # Mapa – Heatmap top-down
    st.subheader("Mapa de calor das reclamações (top-down)")
    st.caption("Ajuste filtros no painel esquerdo.")
    if len(f) > 0:
        midpoint = (np.average(f["lat"]), np.average(f["lon"]))
        layer = pdk.Layer(
            "HeatmapLayer",
            data=f[["lat","lon"]],
            get_position='[lon, lat]',
            aggregation='SUM',
            get_weight=1,
            radiusPixels=55
        )
        r = pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(
                latitude=midpoint[0], longitude=midpoint[1],
                zoom=4.2, pitch=0, bearing=0
            ),
            layers=[layer],
            tooltip={"text":"Densidade de demandas"}
        )
        st.pydeck_chart(r, use_container_width=True)
    else:
        st.info("Sem dados para os filtros selecionados.")

    st.markdown("---")

    # Canal / Origem e Série temporal
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Volume por canal (Ouvidoria)")
        canal_counts = f["canal"].value_counts().reset_index()
        canal_counts.columns = ["canal","qtd"]
        fig = px.bar(canal_counts, x="canal", y="qtd", text_auto=True, template="plotly_white",
                     labels={"qtd":"Demandas","canal":"Canal"})
        fig.update_traces(marker_line_color=PRIMARY, marker_line_width=1.5)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Volume por origem (Produto/Serviço)")
        origem_counts = f["produto_origem"].value_counts().reset_index()
        origem_counts.columns = ["produto_origem","qtd"]
        fig0 = px.bar(origem_counts, x="produto_origem", y="qtd", text_auto=True, template="plotly_white",
                      labels={"qtd":"Demandas","produto_origem":"Origem"})
        fig0.update_traces(marker_line_color=PRIMARY, marker_line_width=1.5)
        st.plotly_chart(fig0, use_container_width=True)

    with c2:
        st.subheader("Série temporal (diária)")
        ts = f.set_index("timestamp").resample("D").size().rename("qtd").reset_index()
        fig2 = px.line(ts, x="timestamp", y="qtd", markers=True, template="plotly_white",
                       labels={"timestamp":"Data","qtd":"Demandas"})
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    # Sankey (Origem → Canal → Triagem → Fila → Desfecho)
    st.subheader("Fluxo de atendimento (Sankey)")
    st.caption("Origem (Produto/Serviço) → Canal (Ouvidoria) → Triagem → Fila → Desfecho")
    if len(f) > 0:
        stages = ["produto_origem","canal","etapa2","etapa3","desfecho"]
        nodes = pd.Index(pd.unique(f[stages].values.ravel())).tolist()
        node_index = {n:i for i,n in enumerate(nodes)}
        links = []
        for a,b in zip(stages[:-1], stages[1:]):
            ct = f.groupby([a,b]).size().reset_index(name="value")
            for _, row in ct.iterrows():
                links.append({"source": node_index[row[a]], "target": node_index[row[b]], "value": int(row["value"])})
        fig3 = go.Figure(data=[go.Sankey(
            node=dict(pad=20, thickness=16, line=dict(width=0.5, color="gray"), label=[str(n) for n in nodes]),
            link=dict(source=[l["source"] for l in links], target=[l["target"] for l in links], value=[l["value"] for l in links])
        )])
        fig3.update_layout(template="plotly_white")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Sem dados para montar o fluxo.")

    st.markdown("---")

    # Matriz Impacto x Esforço
    st.subheader("Matriz de priorização (Impacto × Esforço)")
    initiatives = pd.DataFrame({
        "iniciativa": [
            "Classificação automática de e-mails","Roteamento por tópico","IVR inteligente (URA)",
            "Chatbot de triagem","Callback proativo para picos","Base de conhecimento dinâmica"
        ],
        "impacto": [0.85, 0.75, 0.65, 0.60, 0.55, 0.70],
        "esforco": [0.35, 0.40, 0.55, 0.45, 0.30, 0.50]
    })
    fig4 = px.scatter(initiatives, x="esforco", y="impacto", text="iniciativa",
                      range_x=[0,1], range_y=[0,1], template="plotly_white",
                      labels={"esforco":"Esforço (0-1)","impacto":"Impacto (0-1)"})
    fig4.update_traces(textposition="top center")
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")

    # Amostra de registros (com novas colunas)
    st.subheader("Amostra de registros")
    cols = ["timestamp","produto_origem","motivo_contato","canal_previo","tem_protocolo_previo",
            "canal","topico","severidade","uf","regiao","fcr","sla_ok","sentimento","cidade","lat","lon"]
    st.dataframe(f[cols].sample(min(500, len(f))).sort_values("timestamp", ascending=False) if len(f)>0 else f)
