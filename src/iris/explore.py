from sklearn.datasets import load_iris



def explore_dataset():
    iris = load_iris()

    X = iris.data
    y = iris.target

    print("=== DATASET IRIS ===")

    print("\nDimensions des données :", X.shape)
    print("Dimensions de la cible :", y.shape)

    print("\nCaractéristiques :")
    for feature in iris.feature_names:
        print("-", feature)

    print("\nEspèces :")
    for species in iris.target_names:
        print("-", species)

    print("\nPremières observations :")
    print(X[:5])

    print("\nPremières étiquettes :")
    print(y[:5])


if __name__ == "__main__":
    explore_dataset()
