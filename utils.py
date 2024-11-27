import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def process_data(data):
    stats = data.describe(include='all').to_dict()
    os.makedirs('static/graphs', exist_ok=True)
    graphs = {}

    # Gráfico de Distribuição do Preço de Compra
    plt.figure(figsize=(12, 8))  # Tamanho do gráfico
    sns.histplot(data['buy_price'].dropna(), kde=True, color='teal', bins=30)  # Histograma com KDE
    plt.xlabel('Preço de Compra (€)', fontsize=16)
    plt.ylabel('Frequência', fontsize=16)
    plt.title('Distribuição do Preço de Compra', fontsize=18)

    # Adicionar informações de média e mediana
    mean_price = data['buy_price'].mean()
    median_price = data['buy_price'].median()
    plt.axvline(mean_price, color='red', linestyle='--', linewidth=1, label=f'Média: €{mean_price:.2f}')
    plt.axvline(median_price, color='green', linestyle='--', linewidth=1, label=f'Mediana: €{median_price:.2f}')

    # Legenda do gráfico
    plt.legend(fontsize=12)
    # Grid
    plt.grid(True, linestyle='--', alpha=0.7)

    # Ajuste dos ticks
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Salvar o gráfico
    buy_price_path = 'static/graphs/dist_price_enhanced.png'
    plt.savefig(buy_price_path)
    plt.close()

    graphs['Distribuição do Preço de Compra (Aprimorada)'] = buy_price_path

    # Gráfico de Distribuição da Área Construída
    plt.figure(figsize=(12, 8))  # Ajustando o tamanho do gráfico
    sns.histplot(data['sq_mt_built'].dropna(), bins=30, color='#4C72B0', kde=True)  # Cor mais suave e linha KDE
    plt.title('Distribuição da Área Construída', fontsize=16)
    plt.xlabel('Área Construída (m²)', fontsize=14)
    plt.ylabel('Quantidade', fontsize=14)

    # Adicionando média e mediana
    mean_val = data['sq_mt_built'].mean()
    median_val = data['sq_mt_built'].median()

    # Adicionando linhas de média e mediana
    plt.axvline(mean_val, color='r', linestyle='--', label=f'Média: {mean_val:.2f} m²')
    plt.axvline(median_val, color='g', linestyle='-', label=f'Mediana: {median_val:.2f} m²')

    # Exibindo a legenda
    plt.legend()

    # Ajustes de visualização
    plt.grid(True, linestyle='--', alpha=0.6)  # Grade suave
    plt.xticks(rotation=45)  # Rotacionando os rótulos do eixo x para melhor visualização
    plt.tight_layout()  # Ajustando layout para melhor aproveitamento do espaço

    # Salvando o gráfico com o caminho correto
    built_area_path = 'static/graphs/dist_sq_mt_built_enhanced.png'
    plt.savefig(built_area_path)
    plt.close()

    graphs['Distribuição da Área Construída'] = built_area_path
    # Gráfico de Distribuição do Número de Quartos
    plt.figure()
    data['n_rooms'].value_counts().plot(kind='bar')
    plt.xlabel('Número de Quartos')
    plt.ylabel('Contagem')
    plt.title('Distribuição do Número de Quartos')
    room_distribution_path = 'static/graphs/room_distribution.png'
    plt.savefig(room_distribution_path)
    plt.close()
    graphs['Distribuição do Número de Quartos'] = room_distribution_path

    # Gráfico de Pizza: Distribuição do Número de Quartos
    plt.figure()
    data['n_rooms'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.ylabel('')
    plt.title('Distribuição do Número de Quartos')
    room_distribution_pie_path = 'static/graphs/room_distribution_pie.png'
    plt.savefig(room_distribution_pie_path)
    plt.close()
    graphs['Distribuição do Número de Quartos (Pizza)'] = room_distribution_pie_path

    # Boxplot: Preço de Compra por Número de Quartos
    plt.figure()
    sns.boxplot(x='n_rooms', y='buy_price', data=data)
    plt.xlabel('Número de Quartos')
    plt.ylabel('Preço de Compra')
    plt.title('Boxplot do Preço de Compra por Número de Quartos')
    boxplot_path = 'static/graphs/price_boxplot.png'
    plt.savefig(boxplot_path)
    plt.close()
    graphs['Boxplot do Preço de Compra por Número de Quartos'] = boxplot_path

    # Heatmap: Correlação entre Variáveis
    plt.figure(figsize=(10, 8))  # Ajustando o tamanho do gráfico
    # Filtrando apenas colunas numéricas
    numeric_data = data.select_dtypes(include=[float, int])
    correlation_matrix = numeric_data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Mapa de Calor das Correlações')
    heatmap_path = 'static/graphs/correlation_heatmap.png'
    plt.savefig(heatmap_path)
    plt.close()
    graphs['Mapa de Calor das Correlações'] = heatmap_path

    return stats, graphs

import folium

def generate_map(data):
    if 'latitude' not in data.columns or 'longitude' not in data.columns:
        return None, "Dados de localização (latitude e longitude) não encontrados no conjunto de dados."

    data = data.dropna(subset=['latitude', 'longitude'])


    if data.empty:
        return None, "Não há dados válidos de localização para gerar o mapa."

    map_center = [data['latitude'].mean(), data['longitude'].mean()]
    m = folium.Map(location=map_center, zoom_start=12)

    for _, row in data.iterrows():
        folium.Marker([row['latitude'], row['longitude']],
                      popup=f"Preço: R$ {row['buy_price']}, Quartos: {row['n_rooms']}").add_to(m)

    map_path = 'static/graphs/map.html'
    m.save(map_path)
    return map_path, None

import plotly.express as px

def generate_interactive_plots(data):
    graphs = {}

    if 'sq_mt_built' in data.columns and 'buy_price' in data.columns and 'n_rooms' in data.columns:
        # Criando o gráfico de dispersão original (Preço x Área Construída x Quartos)
        fig1 = px.scatter(data,
                          x='sq_mt_built',
                          y='buy_price',
                          size='n_rooms',
                          color='buy_price',
                          labels={'sq_mt_built': 'Área Construída (m²)', 'buy_price': 'Preço de Compra',
                                  'n_rooms': 'Número de Quartos'},
                          hover_data={'sq_mt_built': True, 'buy_price': True, 'n_rooms': True},
                          color_continuous_scale='Viridis',
                          title="Gráfico de Dispersão 'Preço x Área Construída x Quartos'")

        plot_path1 = 'static/graphs/interactive_price_area_rooms.html'
        fig1.write_html(plot_path1)
        graphs['Gráfico de Dispersão: Preço x Área Construída x Quartos'] = plot_path1

    if 'sq_mt_useful' in data.columns and 'sq_mt_built' in data.columns and 'n_rooms' in data.columns:
        # Criando o novo gráfico de dispersão (Área Útil x Área Construída x Quartos)
        fig2 = px.scatter(data,
                          x='sq_mt_useful',
                          y='sq_mt_built',
                          size='n_rooms',
                          color='sq_mt_built',
                          labels={'sq_mt_useful': 'Área Útil (m²)', 'sq_mt_built': 'Área Construída (m²)',
                                  'n_rooms': 'Número de Quartos'},
                          hover_data={'sq_mt_useful': True, 'sq_mt_built': True, 'n_rooms': True},
                          color_continuous_scale='Viridis',
                          title="Gráfico de Dispersão 'Área Útil x Área Construída x Quartos'")

        plot_path2 = 'static/graphs/interactive_useful_built_rooms.html'
        fig2.write_html(plot_path2)
        graphs['Gráfico de Dispersão: Área Útil x Área Construída x Quartos'] = plot_path2

    return graphs
