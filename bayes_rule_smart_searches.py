import sys, random
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
        c = coords
        coords = [coord for coord in coords if coord not in coords_del]
        if len(coords) == 0:
            coords = 1
        else:
            random.shuffle(coords)
            coords = coords[:int(len(c) * effectiveness_prob)]
        loc_actual = (self.sailor_actual[0], self.sailor_actual[1])
        if area_num == self.area_actual and loc_actual in coords:
            return f'Found sailor in Search Area {area_num}!', coords
        else:
            return 'Not found', coords

    def revise_target_probs(self):
        """Update area target probabilities based on search effectiveness."""
        denom = self.p1 * (1 - self.sep1) + self.p2 * (1 - self.sep2) + self.p3 * (1 - self.sep3)
        self.p1 = self.p1 * (1 - self.sep1) / denom
        self.p2 = self.p2 * (1 - self.sep2) / denom
        self.p3 = self.p3 * (1 - self.sep3) / denom


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


def main():
    app = Search('Cape_Python')
    app.draw_map(last_known=(160, 290))
    sailor_x, sailor_y = app.sailor_final_location(num_search_areas=3)
    print('-' * 65)
    print('\nInitial Target (P) Probabilities:')
    print(f"P1 = {app.p1}, P2 = {app.p2}, P3 = {app.p3}")
    search_num = 1
    coords_1, coords_2, coords_3 = [], [], []
    app.sep1, app.sep2, app.sep3 = 0, 0, 0
    prev_sep1, prev_sep2, prev_sep3 = 0, 0, 0
    while True:
        app.calc_search_effectiveness()
        draw_menu(search_num)
        choice = input('Enter choice: ')

        if choice == '0':
            sys.exit()
        elif choice == '1':
            results_1, coords_1 = app.conduct_search(1, app.sa1, app.sep1, coords_1)
            coords = coords_1
            results_2, coords_1 = app.conduct_search(1, app.sa1, app.sep1, coords_1)
            if coords_1 == 1:
                app.sep1 = 1.0
            else:
                app.sep1 = (len(set(coords + coords_1))) / (len(app.sa1)**2)
            prev_sep1 = app.sep1
            app.sep2 = prev_sep2
            app.sep3 = prev_sep3
        elif choice == '2':
            results_1, coords_2 = app.conduct_search(2, app.sa2, app.sep2, coords_2)
            coords = coords_2
            results_2, coords_2 = app.conduct_search(2, app.sa2, app.sep2, coords_2)
            if coords_2 == 1:
                app.sep2 = 1.0
            else:
                app.sep2 = (len(set(coords + coords_2))) / (len(app.sa2) ** 2)
            prev_sep2 = app.sep2
            app.sep1 = prev_sep1
            app.sep3 = prev_sep3
        elif choice == '3':
            results_1, coords_3 = app.conduct_search(3, app.sa3, app.sep3, coords_3)
            coords = coords_3
            results_2, coords_3 = app.conduct_search(3, app.sa3, app.sep3, coords_3)
            if coords_3 == 1:
                app.sep3 = 1.0
            else:
                app.sep3 = (len(set(coords + coords_3))) / (len(app.sa3) ** 2)
            prev_sep3 = app.sep3
            app.sep1 = prev_sep1
            app.sep2 = prev_sep2
        elif choice == '4':
            results_1, coords_1 = app.conduct_search(1, app.sa1, app.sep1, coords_1)
            if coords_1 == 1:
                app.sep1 = 1.0
            results_2, coords_2 = app.conduct_search(2, app.sa2, app.sep2, coords_2)
            if coords_2 == 1:
                app.sep2 = 1.0
            prev_sep1 = app.sep1
            prev_sep2 = app.sep2
            app.sep3 = prev_sep3
        elif choice == '5':
            results_1, coords_1 = app.conduct_search(1, app.sa1, app.sep1, coords_1)
            if coords_1 == 1:
                app.sep1 = 1.0
            results_2, coords_3 = app.conduct_search(3, app.sa3, app.sep3, coords_3)
            if coords_3 == 1:
                app.sep3 = 1.0
            prev_sep1 = app.sep1
            prev_sep3 = app.sep3
            app.sep2 = prev_sep2
        elif choice == '6':
            results_1, coords_2 = app.conduct_search(2, app.sa2, app.sep2, coords_2)
            if coords_2 == 1:
                app.sep2 = 1.0
            results_2, coords_3 = app.conduct_search(3, app.sa3, app.sep3, coords_3)
            if coords_3 == 1:
                app.sep1 = 1.0
            app.sep1 = prev_sep1
            prev_sep2 = app.sep2
            prev_sep3 = app.sep3
        elif choice == '7':
            main()
        else:
            print('Invalid choice. Try again.', file=sys.stderr)
            continue

        app.revise_target_probs()  # Use BAYES' RULE to update target probabilities

        print(f"\nSearch {search_num} Results 1 = {results_1}", file=sys.stderr)
        print(f"Search {search_num} Results 2 = {results_2}\n", file=sys.stderr)
        print(f"Search {search_num} Effectiveness (E):")
        print(f"E1 = {app.sep1}, E2 = {app.sep2}, E3 = {app.sep3}")

        if results_1 == 'Not found' and results_2 == 'Not found':
            print(f'New Target Probabilities (P) for Search {search_num + 1}:')
            print(f"P1 = {app.p1}, P2 = {app.p2}, P3 = {app.p3}")
        else:
            cv.circle(app.img, (int(sailor_x), int(sailor_y)), 3, (255, 0, 0), -1)
            cv.imshow('Search Area', app.img)
            cv.waitKey(1500)
            main()
        search_num += 1


if __name__ == '__main__':
    main()
