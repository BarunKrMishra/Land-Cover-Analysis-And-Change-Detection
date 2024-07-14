import tkinter as tk
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def get_user_input(year):
    try:
        resident_area = float(resident_area_entries[year].get())
        agriculture_area = float(agriculture_area_entries[year].get())
        barren_lands_area = float(barren_lands_area_entries[year].get())
        data[year] = [resident_area, agriculture_area, barren_lands_area]
    except ValueError:
        messagebox.showerror("Error", "Invalid input. Please enter numeric values.")

def predict_future(years, values, future_years):
    model = LinearRegression()
    model.fit(np.array(years).reshape(-1, 1), values)
    predictions = model.predict(np.array(future_years).reshape(-1, 1))
    return model, predictions

def plot_charts():
    for year in years:
        get_user_input(year)

    for year in years:
        values = data[year]
        plt.figure(figsize=(10, 6))
        plt.bar(categories, values, color=['blue', 'green', 'brown'])
        plt.title(f'Land Use Distribution - Bar Chart ({year})')
        plt.xlabel('Category')
        plt.ylabel('Area (sq km)')
        plt.show()

    for year in years:
        values = data[year]
        plt.figure(figsize=(10, 6))
        plt.pie(values, labels=categories, autopct='%1.1f%%', colors=['blue', 'green', 'brown'])
        plt.title(f'Land Use Distribution - Pie Chart ({year})')
        plt.show()

    prediction_output = ''
    for i, category in enumerate(categories):
        values = [data[year][i] for year in years]
        model, predictions = predict_future(years, values, future_years)

        plt.figure(figsize=(10, 6))
        plt.plot(years, values, marker='o', linestyle='-', label=f'{category} (Actual)')
        plt.plot(future_years, predictions, marker='x', linestyle='--', label=f'{category} (Predicted)')

        plt.title(f'Land Use Distribution - Line Chart ({category})')
        plt.xlabel('Year')
        plt.ylabel('Area (sq km)')
        plt.grid(True)
        plt.legend()
        plt.show()

        prediction_output += f'\n{category} Predictions:'
        for fy, pred in zip(future_years, predictions):
            prediction_output += f'\nPredicted {category} for {fy}: {pred:.2f} sq km'

        y_pred_train = model.predict(np.array(years).reshape(-1, 1))
        accuracy = r2_score(values, y_pred_train)
        prediction_output += f"\nAccuracy (R-squared) score for {category}: {accuracy:.2f}\n"

    # Show output in a new popup window
    popup = tk.Tk()
    popup.title("Predictions and Accuracy")
    popup.geometry("1920x1080")

    output_text = ScrolledText(popup, width=60, height=30, font=("Helvetica", 12))
    output_text.insert(tk.END, prediction_output)
    output_text.pack()

    popup.mainloop()

root = tk.Tk()
root.title("Land Use Distribution Prediction")

data = {}
years = [2009, 2014, 2019, 2023]
future_years = [2024, 2025, 2026, 2027]
categories = ['Resident Area', 'Agriculture Area', 'Barren Lands Area']

resident_area_entries = {}
agriculture_area_entries = {}
barren_lands_area_entries = {}

for year in years:
    frame = tk.Frame(root)
    frame.pack()

    tk.Label(frame, text=f"Year {year}", font=("Helvetica", 14)).grid(row=0, column=0, padx=5, pady=5)

    resident_area_label = tk.Label(frame, text="Resident Area (sq km):", font=("Helvetica", 12))
    resident_area_label.grid(row=1, column=0, padx=5, pady=5)
    resident_area_entry = tk.Entry(frame, font=("Helvetica", 12))
    resident_area_entry.grid(row=1, column=1, padx=5, pady=5)
    resident_area_entries[year] = resident_area_entry

    agriculture_area_label = tk.Label(frame, text="Agriculture Area (sq km):", font=("Helvetica", 12))
    agriculture_area_label.grid(row=2, column=0, padx=5, pady=5)
    agriculture_area_entry = tk.Entry(frame, font=("Helvetica", 12))
    agriculture_area_entry.grid(row=2, column=1, padx=5, pady=5)
    agriculture_area_entries[year] = agriculture_area_entry

    barren_lands_area_label = tk.Label(frame, text="Barren Lands Area (sq km):", font=("Helvetica", 12))
    barren_lands_area_label.grid(row=3, column=0, padx=5, pady=5)
    barren_lands_area_entry = tk.Entry(frame, font=("Helvetica", 12))
    barren_lands_area_entry.grid(row=3, column=1, padx=5, pady=5)
    barren_lands_area_entries[year] = barren_lands_area_entry

    tk.Label(frame, text="").grid(row=4, column=0)  # Spacer

tk.Button(root, text="Plot Charts", command=plot_charts).pack()

root.mainloop()