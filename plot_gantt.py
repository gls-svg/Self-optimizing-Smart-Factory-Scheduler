# plot_gantt.py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_gantt(schedule_csv, title="Schedule"):
    df = pd.read_csv(schedule_csv)
    machines = sorted(df['machine'].unique())
    jobs = df['job'].unique()
    colors = plt.cm.tab20(range(len(jobs)))
    job_color = {j: colors[i] for i, j in enumerate(jobs)}

    fig, ax = plt.subplots(figsize=(14, 6))
    for _, row in df.iterrows():
        m_idx = machines.index(row['machine'])
        ax.barh(m_idx, row['end'] - row['start'], left=row['start'],
                color=job_color[row['job']], edgecolor='black', height=0.6)
        ax.text(row['start'] + (row['end'] - row['start'])/2, m_idx,
                row['job'].split('_')[1], ha='center', va='center', fontsize=6)

    ax.set_yticks(range(len(machines)))
    ax.set_yticklabels([f"M{m}" for m in machines])
    ax.set_xlabel("Time")
    ax.set_title(title)
    patches = [mpatches.Patch(color=job_color[j], label=j) for j in jobs]
    ax.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=7)
    plt.tight_layout()
    plt.savefig(schedule_csv.replace(".csv", "_gantt.png"), dpi=150)
    plt.show()
    print(f"Saved: {schedule_csv.replace('.csv', '_gantt.png')}")

plot_gantt("outputs/rl_schedule.csv", "RL (PPO) Schedule - FT06")
plot_gantt("outputs/fcfs_schedule.csv", "FCFS Schedule - FT06")
plot_gantt("outputs/spt_schedule.csv", "SPT Schedule - FT06")