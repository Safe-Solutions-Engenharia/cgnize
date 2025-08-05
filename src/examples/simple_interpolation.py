from cgnize._climbinterp.adv_interpolation import ClimbInterp
from cgnize._climbinterp.data_arrange import UpperEnvelopeFilter
from random import sample
import matplotlib.pyplot as plt


def main() -> None:
    x_values = sample(range(1, 51), 50)
    y_values = sample(range(1, 51), 50)

    arrange_data = UpperEnvelopeFilter(x_values, y_values)
    #random_numbers_x, random_numbers_y = arrange_data.arranged_x, arrange_data.arranged_y

    random_numbers_x = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    random_numbers_y = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


    plt.scatter(random_numbers_x, random_numbers_y)
    plt.show()
    
    ClimbInterp(random_numbers_x, random_numbers_y, show_graph=True)

if __name__ == "__main__":
    main()