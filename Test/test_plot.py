from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

from geo_ita.src._data import get_high_resolution_population_density_df
from geo_ita.src._plot import *
import geo_ita.src.config as cfg
from PIL import Image, ImageChops
from geo_ita.src.definition import *

import unittest


def compare_images(new_image_path, test_image_path):
    """

    """
    if Path(test_image_path).exists():
        img1 = Image.open(test_image_path).convert("RGB")
        img2 = Image.open(new_image_path).convert("RGB")
        diff = ImageChops.difference(img1, img2)

        if not diff.getbbox():
            # Delete new file if the comparison is successful
            Path(new_image_path).unlink()
        else:
            # Overwrite the old file with the new one and show
            Image.open(new_image_path).show()
            Image.open(test_image_path).show()

            input("Press Enter to continue...")
            Path(new_image_path).replace(test_image_path)
    else:
        Path(new_image_path).replace(test_image_path)
        os.startfile(test_image_path)


def capture_screenshot(html_path, screenshot_path):
    """
    Cattura uno screenshot del file HTML usando Selenium.
    """
    # Configura il driver di Chrome (o un altro browser)
    options = Options()
    options.add_argument("--headless")  # Esegui in modalità headless
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1200,800")

    service = ChromeService()
    driver = webdriver.Chrome(service=service, options=options)

    try:
        # Carica l'HTML locale
        html_url = f"file://{os.path.abspath(html_path)}"
        driver.get(html_url)

        # Attendi che il contenuto sia caricato (se necessario)
        driver.implicitly_wait(5)

        # Trova il grafico (opzionale, se vuoi catturare una parte specifica)
        element = driver.find_element(By.CLASS_NAME, "bk-Row")  # Classe Bokeh
        element.screenshot(screenshot_path)  # Screenshot di un elemento specifico

    finally:
        driver.quit()


def compare_html_files(new_image_path, test_image_path):
    """
    Compare two HTML files.
    """
    if Path(test_image_path).exists():
        capture_screenshot(new_image_path, "new_screenshot.png")
        capture_screenshot(test_image_path, "test_screenshot.png")
        img1 = Image.open("test_screenshot.png").convert("RGB")
        img2 = Image.open("new_screenshot.png").convert("RGB")
        diff = ImageChops.difference(img1, img2)

        diff_array = np.array(diff)
        if np.mean(diff_array) < 1:
            # Delete new file if the comparison is successful
            Path(new_image_path).unlink()
            return
        diff.save("diff.png")
        # Overwrite the old file with the new one and show
        os.startfile(test_image_path)
        os.startfile(new_image_path)
        input("Press Enter to continue...")
        Path(new_image_path).replace(test_image_path)
    else:
        Path(new_image_path).replace(test_image_path)
        os.startfile(test_image_path)


class TestPlot(unittest.TestCase):

    # plot_choropleth_map

    def test_plot_choropleth_map(self):
        tests = [
           (GeoLevel.REGIONE, plot_choropleth_map_regionale, plot_choropleth_map_regionale_interactive),
           (GeoLevel.PROVINCIA, plot_choropleth_map_provinciale, plot_choropleth_map_provinciale_interactive),
            (GeoLevel.COMUNE, plot_choropleth_map_comunale, plot_choropleth_map_comunale_interactive)
        ]
        for level, func, func_interactive in tests:
            df = get_df(level)
            tag = get_tag_registry(CodeLevel.DENOMINATION, level)
            func(df, tag, cfg.TAG_SUPERFICIE, save_path=f"test_choropleth_{level}_new.png", show_plot=False)
            compare_images(f"test_choropleth_{level}_new.png", f"test_choropleth_{level}.png")
            func(df, tag, cfg.TAG_SUPERFICIE, filter_regione="Toscana",
                 save_path=f"test_choropleth_toscana_{level}_new.png", show_plot=False)
            compare_images(f"test_choropleth_toscana_{level}_new.png", f"test_choropleth_toscana_{level}.png")

            values = {
                cfg.TAG_SUPERFICIE: "Superficie",
                cfg.TAG_POPOLAZIONE: "Popolazione",
            }
            func_interactive(df, tag, values, save_path=f"test_choropleth_interactive_{level}_new.html", show_plot=False)
            compare_html_files(f"test_choropleth_interactive_{level}_new.html", f"test_choropleth_interactive_{level}.html")
            func_interactive(df, tag, values, filter_regione="Toscana",
                             save_path=f"test_choropleth_interactive_toscana_{level}_new.html", show_plot=False)
            compare_html_files(f"test_choropleth_interactive_toscana_{level}_new.html", f"test_choropleth_interactive_toscana_{level}.html")

    def test_plot_point_map(self):
        df = get_df(GeoLevel.PROVINCIA)
        df = gpd.GeoDataFrame(df, geometry="geometry")
        df["points"] = df.sample_points(size=10, seed=42)
        df["geometry"] = df["points"]
        df.drop(columns=["points"], inplace=True)
        df = df.explode("geometry")
        df = df.reset_index(drop=True)
        plot_point_map(df, color_column=cfg.TAG_SUPERFICIE, save_path="test_point_map_new.png", show_plot=False)
        compare_images("test_point_map_new.png", "test_point_map_test.png")

        plot_point_map(df, filter_regione="Toscana", color_column=cfg.TAG_SUPERFICIE, add_map_background=False,
                       save_path="test_point_map_toscana_new.png", show_plot=False)
        compare_images("test_point_map_toscana_new.png", "test_point_map_toscana_test.png")

        plot_point_map_interactive(df, color_column=cfg.TAG_SUPERFICIE, save_path="test_point_map_interactive_new.html",
                                   show_plot=False)
        compare_html_files("test_point_map_interactive_new.html", "test_point_map_interactive_test.html")

        plot_point_map_interactive(df, color_column=cfg.TAG_SUPERFICIE, filter_regione="Toscana",
                                   save_path="test_point_map_interactive_toscana_new.html", show_plot=False)
        compare_html_files("test_point_map_interactive_toscana_new.html", "test_point_map_interactive_toscana_test.html")

    def test_plot_density_map(self):
        """test_df = get_df(GeoLevel.COMUNE)
        plot_density_map(test_df, latitude_column='center_y', longitude_column='center_x',
                         save_path="test_plot_density_map_new.png")
        compare_images("test_plot_density_map_new.png", "test_plot_density_map_test.png")

        plot_density_map(test_df, color_column="popolazione", latitude_column='center_y', longitude_column='center_x',
                         save_path="test_plot_density_map_toscana_new.png", filter_regione="Toscana")
        compare_images("test_plot_density_map_toscana_new.png", "test_plot_density_map_toscana_test.png")
        """
        test_df = get_high_resolution_population_density_df()

        plot_density_map(test_df, latitude_column='Lat', longitude_column='Lon', color_column="Population",
                         save_path="test_plot_density_map_high_res_new.png", filter_regione="Toscana")
        compare_images("test_plot_density_map_high_res_new.png", "test_plot_density_map_high_res_test.png")


