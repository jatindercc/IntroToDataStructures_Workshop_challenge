import pandas as pd
import matplotlib.pyplot as plt

from matplotlib_venn import venn3
from statistics import mode, StatisticsError


class StatsCalculation:
    def __init__(self, csv_path=None, data=None):
        if csv_path:
            self.df = pd.read_csv(csv_path)
        elif data:
            self.df = pd.DataFrame(data)
        else:
            raise ValueError("Provide either a CSV path or a data dictionary.")

    def get_numeric_columns(self):
        return self.df.select_dtypes(include='number').columns.tolist()

    def calculate_mean(self, colName=None):
        if colName:
            return self.df[colName].mean()
        return self.df.mean(numeric_only=True)

    def calculate_median(self, colName=None):
        if colName:
            return self.df[colName].median()
        return self.df.median(numeric_only=True)

    def calculate_mode(self, colName=None):
        if colName:
            return self.df[colName].mode()
        return self.df.mode()
    def calculate_mode(self, colName=None):
        if colName:
            try:
                return mode(self.df[colName].dropna())
            except StatisticsError:
                return   "No unique mode"

    def summary_table(self):
        mean = self.calculate_mean()
        median = self.calculate_median()
        mode_ = self.calculate_mode()
        summary_df = pd.DataFrame({
            "Mean": mean,
            "Median": median,
            "Mode": mode_
        })
        return summary_df
    
    def varianceMethod(self):
        # Variance
        variance = self.df.var(numeric_only=True)

        # Standard Deviation
        std_dev = self.df.std(numeric_only=True)

        # Quartiles (Q1, Q2, Q3) and Min/Max
        quartiles = self.df.quantile([0.0, 0.25, 0.5, 0.75, 1.0], numeric_only=True)


        # Transpose quartiles for better formatting
        quartiles_transposed = quartiles.T
        quartiles_transposed.columns = ['Min', 'Q1', 'Q2 (Median)', 'Q3', 'Max']

        # Combine all stats
        all_stats = pd.concat([variance.rename("Variance"),
                                std_dev.rename("Std Dev"),
                                quartiles_transposed], axis=1)
        # Round for clarity
        all_stats = all_stats.round(2)
        return all_stats

    def boxPlotData(self, columns=None):
        # Get numeric column names as a list
        numeric_cols = self.df.select_dtypes(include='number').columns.tolist()

        # If specific columns are provided, use them
        if columns:
            numeric_cols = [col for col in columns if col in numeric_cols]
        if not numeric_cols:
            raise ValueError("No numeric columns available for box plot.")  
        # Create box plot
        plt.figure(figsize=(10, 6))
        self.df.boxplot(column=numeric_cols, grid=False)
        plt.title("Box Plot of Titanic Data Numeric Features")
        plt.ylabel("Value Range")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def vennDiagramData(self,  col1='Age', col2='Fare', col3='Survived',  threshold1=30, threshold2=20):
        # Define sets
        young = set(self.df[self.df[col1] < threshold1].index)
        high_fare = set(self.df[self.df[col2] > threshold2].index)
        survived = set(self.df[self.df[col3] == 1].index)

        plt.figure(figsize=(8, 6))
        venn3([young, high_fare, survived], (f'{col1} < {threshold1}', f'{col2} > {threshold2}', f'{col3} == 1'))
        plt.title(f"Venn Diagram: Overlap of {col1}, {col2}, and {col3}")
        plt.show()

    def scatterPlotData(self, col1='Age', col2='Fare'):
        # Scatter Plot: Age vs. Fare
        plt.figure(figsize=(10, 6))
        plt.scatter(self.df[col1], self.df[col2], alpha=0.5)
        plt.title(f"Scatter Plot of {col1} vs. {col2}")
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.grid(True)
        plt.show()
    
    def histogramData(self, column='Age', bins=30):
        # Histogram: Age Distribution
        plt.figure(figsize=(10, 6))
        self.df[column].hist(bins=bins, edgecolor='black', alpha=0.7)
        plt.title(f"Histogram of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.grid(False)
        plt.show()

    def scatterPlotDataSeaborn(self, x_col='Age', y_col='Fare', hue_col='Survived'):
        import seaborn as sns
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.df, x=x_col, y=y_col, hue=hue_col, palette='Set1', alpha=0.7)
        plt.title(f"Scatter Plot of {x_col} vs. {y_col} colored by {hue_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.grid(True)
        plt.show() 
    
    def histogramDataSeaborn(self, column='Age', bins=30, hue_col='Survived'):
        import seaborn as sns
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.df, x=column, bins=bins, hue=hue_col, multiple="stack", edgecolor='black', alpha=0.7)
        plt.title(f"Histogram of {column} colored by {hue_col}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.grid(False)
        plt.show()

    def getSummary(self):
        # Generate summary statistics
        summary = self.df.describe(include='all').transpose()
        mean = self.calculate_mean()
        median = self.calculate_median()
        mode_ = self.calculate_mode()
        summary['Mean'] = mean
        summary['Median'] = median
        summary['Mode'] = mode_

        return summary