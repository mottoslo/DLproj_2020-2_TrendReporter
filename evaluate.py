import os
import matplotlib.pyplot as plt
import numpy as np

def evaluate(test, predict, title, plot=True, figsize=(20, 13), s=20, filename=None,FLAGS=None):  # 수정됨
    try:
        test = test[s:]
        ## error
        predict["error"] = predict["clicks"] - predict["forecast"]
        predict["error_pct"] = predict["error"] / predict["clicks"]
        predict["error_rate"] = predict["error_pct"].apply(lambda x: np.abs(x))

        ## plot
        if plot == True:
            fig = plt.figure(figsize=figsize)
            fig.suptitle(title, fontsize=20)
            ax1 = fig.add_subplot(2, 2, 1)
            ax2 = fig.add_subplot(2, 2, 2)
            ax3 = fig.add_subplot(2, 2, 3)
            ax4 = fig.add_subplot(2, 2, 4)
            ### training
            test[["clicks", "forecast"]].plot(
                color=["black", "green"], title="Test", grid=True, ax=ax1)
            ax1.set(xlabel=None)

            ### predict
            predict[["clicks", "forecast"]].plot(
                color=["black", "red"], title="Predict from {}".format(FLAGS.pred_index), grid=True, ax=ax2)
            ax2.set(xlabel=None)

            ### error
            predict[["error_rate"]].plot(ax=ax3, color=["green"], title="error", grid=True)
            ax3.set(xlabel=None)

            ### error distribution
            predict[["error_rate"]].plot(ax=ax4, color=["red"], kind='kde', title="error Distribution",
                                         grid=True)
            ax4.set(ylabel=None)

            # plt.show()
            if not os.path.exists("saved_fig"):
                os.mkdir("saved_fig")
            plt.savefig('saved_fig/{}_fig.png'.format(filename))
            # print("saved figure in saved_fig/{}.png".format(filename))

        return test[["clicks", "forecast"]], predict[["clicks", "forecast", "error", "error_rate"]]

    except Exception as e:
        print("--- got error ---")
        print(e)