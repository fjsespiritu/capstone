def plot_target(df, target: str, figsize=(10, 5), color='green', linewidth=1):

    x = df.index
    y = df[target] 

    plt.figure(figsize=figsize)
    plt.title(f'Monthly {target} (2000 - 2025Q2)', fontsize=14)  
    plt.plot(x, y, color=color, linewidth=linewidth)

    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.xlabel("Year")
    plt.ylabel(target) 
    plt.xticks(rotation=45, fontsize=10)

    plt.tight_layout()
    plt.show()