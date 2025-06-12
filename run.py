from app import create_app, camera

app = create_app()

def main():
    app.run(debug=True)

if __name__ == "__main__":
    main()
