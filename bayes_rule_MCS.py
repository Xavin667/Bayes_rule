"""File version that uses Monte Carlo simulation to choose menu options, based on the best probability
In loop we check if choosing twice the same area each day is better than searching two areas each day"""
import sys
import random
import itertools as it
import cv2 as cv
import numpy as np

MAP_FILE = 'cape.png'

# Define the corners of the three search areas (UL = upper left, LR = lower right)
SA1_CORNERS = (130, 265, 180, 315)  # (UL-X, UL-Y, LR-X, LR-Y)
SA2_CORNERS = (80, 255, 130, 305)  # (UL-X, UL-Y, LR-X, LR-Y)
SA3_CORNERS = (105, 205, 155, 255)  # (UL-X, UL-Y, LR-X, LR-Y)


class Search:
    """Bayesian search & rescue game with 3 search areas."""

    def __init__(self, name):
        self.name = name
        self.img = cv.imread(MAP_FILE, cv.IMREAD_COLOR)
        if self.img is None:
            print('Could not load map file {}'.format(MAP_FILE), file=sys.stderr)
            sys.exit(1)

        self.area_actual = 0
        self.sailor_actual = [0, 0]  # As "local" coords within search area

        self.sa1 = self.img[SA1_CORNERS[1]:SA1_CORNERS[3], SA1_CORNERS[0]:SA1_CORNERS[2]]
        self.sa2 = self.img[SA2_CORNERS[1]:SA2_CORNERS[3], SA2_CORNERS[0]:SA2_CORNERS[2]]
        self.sa3 = self.img[SA3_CORNERS[1]:SA3_CORNERS[3], SA3_CORNERS[0]:SA3_CORNERS[2]]

        self.p1 = 0.2
        self.p2 = 0.5
        self.p3 = 0.3  # Prior probabilities of sailor in each search area

        self.sep1 = 0  # Search effort in each search area
        self.sep2 = 0
        self.sep3 = 0

    def draw_map(self, last_known):
        """Display basemap with scale, last known xy location, search areas"""
        cv.line(self.img, (20, 370), (70, 370), (0, 0, 0), 2)  # draw scale line
        cv.putText(self.img, '0', (8, 370), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
        cv.putText(self.img, '50 Nautical Miles', (71, 370), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))

        # Draw search areas
        cv.rectangle(self.img, (SA1_CORNERS[0], SA1_CORNERS[1]), (SA1_CORNERS[2], SA1_CORNERS[3]), (0, 0, 0), 1)
        cv.putText(self.img, '1', (SA1_CORNERS[0] + 3, SA1_CORNERS[1] + 15), cv.FONT_HERSHEY_PLAIN, 1, 0)
        cv.rectangle(self.img, (SA2_CORNERS[0], SA2_CORNERS[1]), (SA2_CORNERS[2], SA2_CORNERS[3]), (0, 0, 0), 1)
        cv.putText(self.img, '2', (SA2_CORNERS[0] + 3, SA2_CORNERS[1] + 15), cv.FONT_HERSHEY_PLAIN, 1, 0)
        cv.rectangle(self.img, (SA3_CORNERS[0], SA3_CORNERS[1]), (SA3_CORNERS[2], SA3_CORNERS[3]), (0, 0, 0), 1)
        cv.putText(self.img, '3', (SA3_CORNERS[0] + 3, SA3_CORNERS[1] + 15), cv.FONT_HERSHEY_PLAIN, 1, 0)

        cv.putText(self.img, '+', last_known, cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
        cv.putText(self.img, '+ = Last Known Position', (274, 355), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
        cv.putText(self.img, '* = Actual Position', (275, 370), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))

        cv.imshow('Search Area', self.img)
        cv.moveWindow('Search Area', 750, 10)
        cv.waitKey(1000)

    def sailor_final_location(self, num_search_areas):
        """Return the actual x, y location of the missing sailor."""
        # Find sailor coordinates with respect to any Search Area subarray.
        self.sailor_actual[0] = np.random.choice(self.sa1.shape[1], 1)
        self.sailor_actual[1] = np.random.choice(self.sa1.shape[0], 1)

        area = int(random.triangular(1, num_search_areas + 1))

        if area == 1:
            x = self.sailor_actual[0] + SA1_CORNERS[0]
            y = self.sailor_actual[1] + SA1_CORNERS[1]
            self.area_actual = 1
        elif area == 2:
            x = self.sailor_actual[0] + SA2_CORNERS[0]
            y = self.sailor_actual[1] + SA2_CORNERS[1]
            self.area_actual = 2
        elif area == 3:
            x = self.sailor_actual[0] + SA3_CORNERS[0]
            y = self.sailor_actual[1] + SA3_CORNERS[1]
            self.area_actual = 3
        return x, y

    def calc_search_effectiveness(self):
        """Set decimal search effectiveness value per search area."""
        self.sep1 = random.uniform(0.2, 0.9)
        self.sep2 = random.uniform(0.2, 0.9)
        self.sep3 = random.uniform(0.2, 0.9)

    def conduct_search(self, area_num, area_array, effectiveness_prob, coords_del):
        """Return search results and list of searched coordinates."""
        local_y_range = range(area_array.shape[0])
        local_x_range = range(area_array.shape[1])
        coords = list(it.product(local_x_range, local_y_range))
        c = list(it.product(local_x_range, local_y_range))
        coords = [coord for coord in coords if coord not in coords_del]
        if len(coords) == 0:
            coords = 1
        else:
            coords = coords[:int(len(c) * effectiveness_prob)]
        loc_actual = (self.sailor_actual[0], self.sailor_actual[1])
        if area_num == self.area_actual and loc_actual in coords:
            return f'Found sailor in Search Area {area_num}!', coords
        else:
            return 'Not found', coords

    def revise_target_probs(self):
        """Update area target probabilities based on search effectiveness."""
        denom = self.p1 * (1 - self.sep1) + self.p2 * (1 - self.sep2) + self.p3 * (1 - self.sep3)
        if denom != 0:
            self.p1 = self.p1 * (1 - self.sep1) / denom
            self.p2 = self.p2 * (1 - self.sep2) / denom
            self.p3 = self.p3 * (1 - self.sep3) / denom
        if denom == 0:
            self.p1 = 1
            self.p2 = 1
            self.p3 = 1
    def reset_target_probs(self):
        """Reset area target probabilities."""
        self.p1 = 0.2
        self.p2 = 0.5
        self.p3 = 0.3

def draw_menu(search_num):
    """Print menu of choices for conducting area searches."""
    print(f'\nSearch {search_num}')
    print(
        """
        Choose next areas to search:
        
        0 - Quit
        1 - Search Area 1 twice
        2 - Search Area 2 twice
        3 - Search Area 3 twice
        4 - Search Area 1 & 2
        5 - Search Area 1 & 3
        6 - Search Area 2 & 3
        7 - Start Over
        """
    )


def monte_carlo_twice(p1, p2, p3):
    """Return the area with the highest probability of containing the sailor."""
    if p1 > p2 and p1 > p3:
        choice = 1
        return choice
    elif p2 > p1 and p2 > p3:
        choice = 2
        return choice
    elif p3 > p1 and p3 > p2:
        choice = 3
        return choice
    elif p1 == p2 and p1 > p3:
        choice = random.choice([1, 2])
        return choice
    elif p1 == p3 and p1 > p2:
        choice = random.choice([1, 3])
        return choice
    elif p2 == p3 and p2 > p1:
        choice = random.choice([2, 3])
        return choice
    elif p1 == p2 == p3:
        choice = random.choice([1, 2, 3])
        return choice


def monte_carlo_once(p1, p2, p3):
    """Return two areas with the highest probability of containing the sailor."""
    if (p1 > p3) and (p2 > p3):
        choice = 4
        return choice
    elif (p1 > p2) and (p3 > p2):
        choice = 5
        return choice
    elif (p2 > p1) and (p3 > p1):
        choice = 6
        return choice
    elif p1 == p2 == p3:
        choice = random.choice([4, 5, 6])
        return choice
    elif p1 == p2 and p1 > p3:
        choice = 4
        return choice
    elif p1 == p3 and p1 > p2:
        choice = 5
        return choice
    elif p2 == p3 and p2 > p1:
        choice = 6
        return choice
    elif p1 == p2 and p1 < p3:
        choice = random.choice([5, 6])
        return choice
    elif p1 == p3 and p1 < p2:
        choice = random.choice([4, 6])
        return choice
    elif p2 == p3 and p2 < p1:
        choice = random.choice([4, 5])
        return choice


search_results = []


def main():
    app = Search('Cape_Python') # Create instance of Search class
    app.draw_map(last_known=(160, 290)) # Draw map with last known location
    sailor_x, sailor_y = app.sailor_final_location(num_search_areas=3) # Generate sailor's final location
    search_num = 1
    i = 1  # Counter for number of searches
    coords_1, coords_2, coords_3 = [], [], [] # List of coordinates searched in each area
    app.sep1, app.sep2, app.sep3 = 0, 0, 0  # Current search effectiveness
    prev_sep1, prev_sep2, prev_sep3 = 0, 0, 0  # Search effectiveness from previous search to remember through the loop
    """
    app.sep1 = prev_sep1  # Set search effectiveness for next search to previous search effectiveness
    """
    while i <= 10000:

        app.calc_search_effectiveness()
        # choice = monte_carlo_twice(app.p1, app.p2, app.p3)  # Choose area to search twice Average: 2.0935
        choice= monte_carlo_once(app.p1, app.p2, app.p3)  # Choose two areas to search once Average: 1.9644

        if choice == 0:
            sys.exit(0)
        elif choice == 1:
            results_1, coords_1 = app.conduct_search(1, app.sa1, app.sep1, coords_1)
            coords = coords_1
            results_2, coords_1 = app.conduct_search(1, app.sa1, app.sep1, coords_1)
            if coords_1 == 1:  # If all coordinates have been searched, set search effectiveness to 1.0
                app.sep1 = 1.0
            else:
                app.sep1 = (len(set(coords + coords_1))) / (len(app.sa1)**2)
            app.sep2 = prev_sep2
            app.sep3 = prev_sep3
        elif choice == 2:
            results_1, coords_2 = app.conduct_search(2, app.sa2, app.sep2, coords_2)
            coords = coords_2
            results_2, coords_2 = app.conduct_search(2, app.sa2, app.sep2, coords_2)
            if coords_2 == 1:
                app.sep2 = 1.0
            else:
                app.sep2 = (len(set(coords + coords_2))) / (len(app.sa2) ** 2)
            app.sep1 = prev_sep1
            app.sep3 = prev_sep3
        elif choice == 3:
            results_1, coords_3 = app.conduct_search(3, app.sa3, app.sep3, coords_3)
            coords = coords_3
            results_2, coords_3 = app.conduct_search(3, app.sa3, app.sep3, coords_3)
            if coords_3 == 1:  # If all coordinates have been searched, set search effectiveness to 1.0
                app.sep3 = 1.0
            else:
                app.sep3 = (len(set(coords + coords_3))) / (len(app.sa3) ** 2)
            app.sep1 = prev_sep1
            app.sep2 = prev_sep2
        elif choice == 4:
            results_1, coords_1 = app.conduct_search(1, app.sa1, app.sep1, coords_1)
            if coords_1 == 1:
                app.sep1 = 1.0
            results_2, coords_2 = app.conduct_search(2, app.sa2, app.sep2, coords_2)
            if coords_2 == 1:
                app.sep2 = 1.0

            app.sep3 = prev_sep3
        elif choice == 5:
            results_1, coords_1 = app.conduct_search(1, app.sa1, app.sep1, coords_1)
            if coords_1 == 1:
                app.sep1 = 1.0
            results_2, coords_3 = app.conduct_search(3, app.sa3, app.sep3, coords_3)
            if coords_3 == 1:
                app.sep3 = 1.0
            app.sep2 = prev_sep2
        elif choice == 6:
            results_1, coords_2 = app.conduct_search(2, app.sa2, app.sep2, coords_2)
            if coords_2 == 1:
                app.sep2 = 1.0
            results_2, coords_3 = app.conduct_search(3, app.sa3, app.sep3, coords_3)
            if coords_3 == 1:
                app.sep3 = 1.0
            app.sep1 = prev_sep1

        if results_1 == 'Not found' and results_2 == 'Not found':
            search_num += 1
            app.revise_target_probs()  # Use BAYES' RULE to update target probabilities

        else:
            search_results.append(search_num)  # Add search number to list
            print(f'I: {i}')
            i += 1
            app = Search('Cape_Python') # Make new instance of Search class
            sailor_x, sailor_y = app.sailor_final_location(num_search_areas=3)  # Generate next sailor's final location
            search_num = 1  # Reset search number
            coords_1, coords_2, coords_3 = [], [], []  # Make coords list empty for next search
            app.sep1, app.sep2, app.sep3 = 0, 0, 0 # Reset search effectiveness
            prev_sep1, prev_sep2, prev_sep3 = 0, 0, 0 # Reset previous search effectiveness
    average = sum(search_results) / len(search_results)  # Calculate average number of searches
    print(f'Average: {average}')
    sys.exit(0)


if __name__ == '__main__':
    main()
