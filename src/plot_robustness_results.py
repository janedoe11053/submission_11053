from ast import literal_eval
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import click


@click.command()
@click.option('-i', '--input-path', 'input_path', required=True)
@click.option('-o', '--output-path', 'output_path', required=True)
def plot_experiments(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    op_df = pd.read_csv(input_path / "results_hx.csv", index_col=0)
    op_df['s'] = op_df['s'].apply(literal_eval)
    op_df = op_df[(op_df.p == 2) | (op_df.cent == 'degree')]

    op_df['p'] = op_df['p'].apply(lambda x: F"p={x}")
    op_df['s'] = op_df['s'].apply(lambda x: F"s=({x[0]},{x[1] - 1})")
    op_df['s'] = op_df['s'].apply(lambda col: col.replace('25', 'n-1'))
    op_df['s'] = op_df['s'].apply(lambda col: col.replace('24', 'n-2'))
    op_df['Params'] = op_df[['cent', 's']].agg('\n'.join, axis=1)

    op_df = op_df.drop(['expl', 'b_id', 'seed', 'cent', 'p', 's'], axis=1)

    sl_df = pd.read_csv(input_path / "results_sl.csv", index_col=0)
    sl_df['c'] = sl_df['c'].apply(lambda x: F"{x}")
    sl_df['Params'] = sl_df[['expl', 'c']].agg('\n'.join, axis=1)
    sl_df = sl_df.drop(['b_id', 'seed', 'expl', 'c'], axis=1)
    op_df = op_df.append(sl_df, ignore_index=True)

    grouped = op_df.groupby(['Params']).median().sort_values(by='L')

    f, ax = plt.subplots(figsize=(20, 5))
    ax.set_yscale("log")

    sns.boxplot(x="Params", y="L", data=op_df, order=grouped.index,
                palette=sns.color_palette("RdBu_r", len(grouped)),
                showfliers=False)
    sns.swarmplot(x="Params", y="L", data=op_df, order=grouped.index, size=4, color=".25")
    plt.tight_layout()
    plt.savefig(output_path / 'boxplot.pdf')


if __name__ == "__main__":
    plot_experiments()
