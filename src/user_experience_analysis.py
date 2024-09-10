import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

#Aggregation Functions
def aggregate_customer_experience(df):
    """
    Aggregate the required customer experience metrics: average TCP retransmission, RTT, throughput, and handset type.
    """
    agg_df = df.groupby('Bearer Id').agg({
        'TCP DL Retrans. Vol (Bytes)': 'mean',
        'Avg RTT DL (ms)': 'mean',
        'Handset Type': 'first',  # Assuming one handset type per customer
        'DL TP < 50 Kbps (%)':'mean',
        '50 Kbps < DL TP < 250 Kbps (%)':'mean',
        '250 Kbps < DL TP < 1 Mbps (%)':'mean',
        'DL TP > 1 Mbps (%)':'mean',
    })
    return agg_df

#Top, Bottom, Frequent Functions
def get_top_bottom_frequent(df, column):
    """
    Compute the top, bottom, and most frequent values for the given column.
    """
    top_10 = df[column].nlargest(10)
    bottom_10 = df[column].nsmallest(10)
    most_frequent = int(df[column].mode().iloc[0])
    
    return top_10, bottom_10, most_frequent

#Visualization Functions
def plot_throughput_distribution(df):
    """
    Plot the distribution of throughput values per handset type using histograms.
    """
    # Define the minimum number of records required for each handset type
    min_records = 10  # Adjust as needed
    
    # Filter out handset types with insufficient data
    sufficient_data_handsets = df['Handset Type'].value_counts()[df['Handset Type'].value_counts() >= min_records].index
    filtered_df = df[df['Handset Type'].isin(sufficient_data_handsets)]
    
    if filtered_df.empty:
        print("No sufficient data available to plot.")
        return

    plt.figure(figsize=(12, 8))
    
    for name, group in filtered_df.groupby('Handset Type'):
        sns.histplot(group[['DL TP < 50 Kbps (%)',
                            '50 Kbps < DL TP < 250 Kbps (%)',
                            '250 Kbps < DL TP < 1 Mbps (%)',
                            'DL TP > 1 Mbps (%)']].values.flatten(),
                     label=name, kde=False, bins=30)
    
    plt.title('Throughput Distribution per Handset Type')
    plt.xlabel('Throughput')
    plt.ylabel('Count')
    plt.legend(title='Handset Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.show()
def plot_tcp_retransmission_per_handset(df):
    """
    Plot the average TCP retransmission per handset type.
    """
    avg_tcp_by_handset = df.groupby('Handset Type')['TCP DL Retrans. Vol (Bytes)'].mean()
    avg_tcp_by_handset.plot(kind='bar')
    plt.title('Average TCP Retransmission per Handset Type')
    plt.xlabel('Handset Type')
    plt.ylabel('Average TCP Retransmission')
    plt.show()

#Clustering Functions
def perform_kmeans_clustering(df, num_clusters=3):
    """
    Perform KMeans clustering on experience metrics.
    """
    kmeans = KMeans(n_clusters=num_clusters)
    features = df['TCP DL Retrans. Vol (Bytes)', 'Avg RTT DL (ms)','DL TP < 50 Kbps (%)']
    df['cluster'] = kmeans.fit_predict(features)
    
    return df, kmeans