import matplotlib.pyplot as plt


def plot_array(values):
    x_axis = range(len(values))
    y_axis = values
    plt.plot(x_axis, y_axis)

    # Skew the x labels
    plt.gcf().autofmt_xdate()

    plt.show()


def main():
    ''' Created for test purposes '''
    # pass


if __name__ == '__main__':
    main()
