import matplotlib.pyplot as plt


def plotFunc(accs, losses, parameter):
    fig, ax = plt.subplots()
    ax.plot(range(len(accs)), accs, 'k', label='accuracy for {}'.format(parameter))
    ax.legend(loc='upper right', shadow=True)
    plt.savefig('./images/{}_acc.png'.format(parameter))
    plt.show()

    fig, bx = plt.subplots()
    bx.plot(range(len(losses)), losses, 'k', label='loss for {}'.format(parameter))
    bx.legend(loc='upper right', shadow=True)
    plt.savefig('./images/{}_loss.png'.format(parameter))
    plt.show()