import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.neighbors import KNeighborsClassifier
temperatures = pd.DataFrame({"country": ["USA", "Canada", "Mexico"], "date": ["2023-01-01", "2023-01-02", "2023-01-03"], "temperature": [30, 20, 25]})
temperature_ind = temperatures.set_index(["country", "date"])
temperature_srt = temperature_ind.sort_index()
result = temperature_srt.loc[("USA", "2023-01-01"), "temperature"]
print (plt.hist(temperature_srt["temperature"], bins=5, color='blue', alpha=0.7))
X = np.array([[30], [20], [25]])
y = np.array([0, 1, 2])  # Labels for USA, Canada, and Mexico
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)

# Generate predictions for all temperatures
temperatures_range = np.arange(0, 100)
predictions = knn.predict(temperatures_range.reshape(-1, 1))

#Using Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=temperatures_range,
    y=predictions,
    mode='markers+lines',
    marker=dict(size=8, color=predictions, colorscale='Viridis', showscale=True),
    line=dict(color='rgba(0,0,0,0.2)'),
    hovertemplate='<b>Temperature:</b> %{x}°F<br><b>Predicted Country:</b> %{y}<extra></extra>'
))
fig.update_layout(
    title='KNN Classification: Temperature to Country',
    xaxis_title='Temperature (°F)',
    yaxis_title='Predicted Country (0=USA, 1=Canada, 2=Mexico)',
    hovermode='closest',
    template='plotly_white'
)
fig.show()

#Using Matplotlib Scatter
plt.figure(figsize=(10, 6))
plt.scatter(temperatures_range, predictions, c=predictions, cmap='viridis', s=50, alpha=0.7, edgecolors='black')
plt.colorbar(label='Predicted Country')
plt.xlabel('Temperature (°F)')
plt.ylabel('Predicted Country (0=USA, 1=Canada, 2=Mexico)')
plt.title('KNN Classification: Temperature to Country')
plt.grid(True, alpha=0.3)
plt.show()
