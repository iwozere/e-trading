# Trading Web UI Startup Guide (Windows)

To start the Web UI module after a restart, follow these steps:

## 1. Quick Start (Recommended)

Run the main development launcher script from the project root:

```batch
bin\trading\trading-bot-webui.bat
```

The script has been updated to be more compatible with various Windows terminals (it now handles UTF-8 correctly and uses plain text if colors aren't supported).

This script will:
- Check your environment and dependencies
- **Initialize/Activate the virtual environment**
- **Start the Backend Server** (available at http://localhost:5003)

> [!NOTE]
> After the backend starts, the script will prompt you to start the frontend in a **separate terminal window**.

## 2. Start the Frontend

Open a **new terminal window** in the project root and run:

```batch
bin\trading\trading-bot-webui.bat --frontend
```

The Frontend will be available at: **http://localhost:5002** (or 5002 depending on config, but the script will tell you).

---

## 🛠️ Alternative Method (Single Command)

If you prefer to use the main Python runner (which tries to launch both):

```batch
.venv\Scripts\python src\web_ui\run_web_ui.py --dev
```

## 🗄️ Database Initialization (If Needed)

If this is the first time you are starting the UI or if you deleted your database:

```batch
.venv\Scripts\python bin\web_ui\init_webui_database.py
```

**Default Credentials:**
- **Admin**: `admin` / `admin`
- **Trader**: `trader` / `trader`
- **Viewer**: `viewer` / `viewer`

---

## 🔍 Access Links

- **Frontend UI**: [http://localhost:5002](http://localhost:5002)
- **Backend API**: [http://localhost:5003](http://localhost:5003)
- **API Documentation**: [http://localhost:5003/docs](http://localhost:5003/docs)
