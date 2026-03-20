
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Imports
import json
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified
from src.data.db.core.database import make_engine

# Import ALL models to ensure foreign keys are resolved
from src.data.db.models.model_users import User, AuthIdentity
from src.data.db.models.model_trading import BotInstance, Trade, Position, PerformanceMetric
from src.data.db.models.model_notification import Message, MessageDeliveryStatus

def migrate():
    print(f"Connecting to database...")
    engine = make_engine()
    
    with Session(engine) as session:
        # Get all bots
        bots = session.execute(select(BotInstance)).scalars().all()
        print(f"Found {len(bots)} bots in database.")
        
        updated_count = 0
        for bot in bots:
            config = bot.config or {}
            changed = False
            bot_name = bot.description or f"Bot {bot.id}"
            
            # 1. Ensure bot_id is present
            if 'bot_id' not in config:
                config['bot_id'] = str(bot.id)
                print(f"  [{bot_name}] Added bot_id: {config['bot_id']}")
                changed = True
            
            # 2. Ensure modules structure is present
            if 'modules' not in config:
                print(f"  [{bot_name}] Converting to modular structure...")
                
                # Extract components
                broker = config.get('broker', {})
                strategy = config.get('strategy', {})
                risk = config.get('risk_management', {})
                notifications = config.get('notifications', {})
                
                # Create modules wrapper
                config['modules'] = {
                    'broker': broker,
                    'strategy': strategy,
                    'risk_management': risk,
                    'notifications': notifications
                }
                
                # Clean up old keys (optional, but makes it truly modular)
                for key in ['broker', 'strategy', 'risk_management', 'notifications']:
                    if key in config:
                        del config[key]
                
                changed = True
            
            if changed:
                bot.config = config
                # Crucial for SQLAlchemy to detect changes in JSON columns
                flag_modified(bot, "config")
                updated_count += 1
        
        if updated_count > 0:
            print(f"Committing changes for {updated_count} bots...")
            session.commit()
            print("Migration successful.")
        else:
            print("No bots required migration.")

if __name__ == "__main__":
    try:
        migrate()
    except Exception as e:
        print(f"Migration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
