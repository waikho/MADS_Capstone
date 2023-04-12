import matplotlib.pyplot as plt
import mplfinance as mpl
import matplotlib.dates as mdates
import pandas as pd

def plot_chart(dollar_bars, name=''):

    """
    plot OHLCV chart
    """
    plot = mpl.plot(dollar_bars, type='candle', volume=True, figratio=(4,1), style='yahoo', title=name)   
    
    return plot


def plot_triple_barrier_events(dollar_bars, events):
    
    plt_temp_df = dollar_bars.copy()
    plt_temp_df.set_index(pd.to_datetime(plt_temp_df.index), inplace=True)

    fig, ax = plt.subplots(figsize=(30,10))
    ax.plot(plt_temp_df.index, plt_temp_df['close'], label='Closing Price')
    for i, event in events.iterrows():
        if not pd.isna(event['ret']):
            left = mdates.date2num(i)
            right = mdates.date2num(event['t1'])
            top = event['close'] * (1 + event['trgt'])
            bottom = event['close'] * (1 - event['trgt'])
            width = right - left #(events_df.t1 - events_df.time).dt.days
            height = top - bottom
            if event['bin'] == 1:
                color = 'blue'
            else:
                color = 'grey'
            ax.add_patch(plt.Rectangle((left, bottom), width, height, fill=True, alpha=0.5, color=color))
            
            # add a label indicating whether the event is long or short
            x = left + width / 2
            y = top + height / 10
            if event['label'] == 1:
                label = 'L'
            else:
                label = 'S'

            # add return percentage under the label
            ret_pct = event['ret'] * 100
            if event['ret'] > 0:
                color = 'green'
            elif event['ret'] < 0:
                color = 'red'
            else:
                color = 'black'
            ax.text(x, y, f"{label}\n{ret_pct:.0f}%", ha='center', va='bottom', fontsize=11, color=color)

    ax.set_xlabel('Date')
    ax.legend()
    plt.show()


def plot_model_metrics(model_metrics, add_msg='between Modeling methods', show_train_test_mean=False):
    for metric in model_metrics.columns:
        ax = model_metrics[metric].plot(kind='barh', rot=0, fontsize=9, color='orange')
        for patch in ax.patches:
            width = patch.get_width()
            height = patch.get_height()
            x, y = patch.get_xy()
            ax.annotate(f'{width:.2f}', (x + width + 0.001, y + height/2), ha='left', va='center')

        if show_train_test_mean == True:
            metric_means = model_metrics.groupby('train_test').mean()
            mean_train = metric_means.loc['Train', metric]
            mean_test = metric_means.loc['Test', metric]
            ax.axvline(x=mean_train, color='red', linestyle='--',alpha=0.7,label=f'Training Mean = {mean_train:.2f}')
            ax.axvline(x=mean_test, color='blue', linestyle='--',alpha=0.7,label=f'Test Mean = {mean_test:.2f}')
            ax.grid(False)
            legend = ax.legend(bbox_to_anchor=(0.9, 0.9), bbox_transform=plt.gcf().transFigure)

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title(f'Comparison of {metric.capitalize()} {add_msg}', fontsize=11)
        plt.show()

def plot_model_metrics_grid(model_metrics, add_msg='', show_train_test_mean=False):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 8), sharey=True)
    metrics = model_metrics.columns
    for i, metric in enumerate(metrics):
        if i < 2:
            ax = axs[i//3, i%3]
        else:
            ax = axs[1, 2]
            ax.set_facecolor('white')
        model_metrics[metric].plot(kind='barh', rot=0, fontsize=9, color='orange', ax=ax)
        for patch in ax.patches:
            width = patch.get_width()
            height = patch.get_height()
            x, y = patch.get_xy()
            ax.annotate(f'{width:.2f}', (x + width + 0.001, y + height/2), ha='left', va='center')

        if show_train_test_mean == True:
            metric_means = model_metrics.groupby('train_test').mean()
            mean_train = metric_means.loc['Train', metric]
            mean_test = metric_means.loc['Test', metric]
            ax.axvline(x=mean_train, color='red', linestyle='--',alpha=0.7,label=f'Training Mean = {mean_train:.2f}')
            ax.axvline(x=mean_test, color='blue', linestyle='--',alpha=0.7,label=f'Test Mean = {mean_test:.2f}')
            ax.grid(False)
            legend = ax.legend(bbox_to_anchor=(0.9, 0.9), bbox_transform=plt.gcf().transFigure)

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title(f'{metric.capitalize()} {add_msg}', fontsize=11)
    plt.tight_layout()
    plt.show()

