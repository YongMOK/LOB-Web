# LOB- Limit order books project

There are three branchs, the first one is the oldes version, second is improvement version and the last one is the newest version.

### Prerequisites

Ensure you have the following installed:
- Python 3.x
- pip (Python package installer)
- virtualenv (optional but recommended)

### Installation

1. **Clone the repository:**
   ```sh
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```sh
   python -m venv venv
   venv\Scripts\activate  # On Windows
   ```

3. **Generate a `requirements.txt` file if it doesn't exist:**
   ```sh
   pip freeze > requirements.txt
   ```

4. **Install the required packages:**
   ```sh
   pip install -r requirements.txt
   ```

5. **Apply database migrations:**
   ```sh
   python manage.py migrate
   ```

6. **Create a superuser for testing:**
   
   Before starting you can create a superuser for testing with our own computer as an admin.
   ```sh
   python manage.py createsuperuser
   ```

8. **Run the development server:**
   ```sh
   python manage.py runserver
   ```
