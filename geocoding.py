from opencage.geocoder import OpenCageGeocode
import pandas as pd
import time

# Use a sua chave de API do OpenCage
key = 'ebf133f9892c480a8b2d4b845f93771e'

geocoder = OpenCageGeocode(key)

def geocode_address(row):
    try:
        if pd.isna(row['raw_address']):
            return pd.Series([None, None])
        results = geocoder.geocode(row['raw_address'])
        if results:
            location = results[0]['geometry']
            time.sleep(1)
            return pd.Series([location['lat'], location['lng']])
    except Exception as e:
        print(f"Erro ao geocodificar {row['raw_address']}: {e}")
    return pd.Series([None, None])

def add_lat_lon(data):
    data['latitude'] = None
    data['longitude'] = None
    total_rows = len(data)
    for idx, row in data.iterrows():
        lat_lon = geocode_address(row)
        data.at[idx, 'latitude'] = lat_lon[0]
        data.at[idx, 'longitude'] = lat_lon[1]
        if idx % 10 == 0:
            print(f"Processados {idx + 1} de {total_rows} endereços")
        if idx % 100 == 0:
            data.to_csv('C:/Users/Paula/Documents/python/trabalho final/projeto/pythonProject2/data/houses_Madrid_parcial.csv', index=False)
            print(f"Progresso salvo após {idx + 1} endereços")
    return data


if __name__ == "__main__":
    data = pd.read_csv('C:/Users/Paula/Documents/python/trabalho final/projeto/pythonProject2/data/houses_Madrid.csv')
    print("Carregando dados...")
    data = add_lat_lon(data)
    print("Salvando dados completos com coordenadas...")
    data.to_csv('C:/Users/Paula/Documents/python/trabalho final/projeto/pythonProject2/data/houses_Madrid_com_lat_lon.csv', index=False)
    print("Processo concluído!")
