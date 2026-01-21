"""Test script for Replicate API."""
import urllib.request
import json
import sys

def test_api(token):
    """Test Replicate API and find available models."""
    
    print("=" * 50)
    print("Replicate API Test")
    print("=" * 50)
    
    # Test 1: Check token validity
    print("\n1. Testing API token...")
    try:
        url = "https://api.replicate.com/v1/models"
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Bearer {token}")
        
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            print("   [OK] API token is valid!")
    except urllib.error.HTTPError as e:
        print(f"   [ERROR] API token invalid: {e.code}")
        return
    except Exception as e:
        print(f"   [ERROR] {e}")
        return
    
    # Test 2: Search for virtual try-on models
    print("\n2. Searching for virtual try-on models...")
    
    models_to_check = [
        "cuuupid/idm-vton",
        "levelsio/neon-tshirt",
        "stability-ai/sdxl",
    ]
    
    working_models = []
    
    for model in models_to_check:
        try:
            owner, name = model.split("/")
            url = f"https://api.replicate.com/v1/models/{owner}/{name}"
            req = urllib.request.Request(url)
            req.add_header("Authorization", f"Bearer {token}")
            
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                
                # Get version
                version = None
                if data.get("latest_version"):
                    version = data["latest_version"].get("id", "")[:12]
                
                print(f"   [OK] {model} - version: {version}...")
                working_models.append((model, data))
                
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"   [NOT FOUND] {model}")
            else:
                print(f"   [ERROR] {model}: {e.code}")
        except Exception as e:
            print(f"   [ERROR] {model}: {e}")
    
    # Show results
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    
    if working_models:
        print(f"\nFound {len(working_models)} accessible model(s):")
        for model, data in working_models:
            latest = data.get("latest_version", {})
            version_id = latest.get("id", "N/A")
            print(f"\n  Model: {model}")
            print(f"  Version: {version_id}")
            print(f"  Description: {data.get('description', 'N/A')[:100]}...")
    else:
        print("\nNo accessible virtual try-on models found.")
        print("You may need to:")
        print("1. Check if your billing is set up at replicate.com")
        print("2. Accept terms for specific models")
    
    return working_models


if __name__ == "__main__":
    print("\nReplicate API Token'inizi girin:")
    print("(Token'i replicate.com/account/api-tokens adresinden alabilirsiniz)\n")
    
    token = input("API Token: ").strip()
    
    if not token:
        print("Token girilmedi!")
        sys.exit(1)
    
    if not token.startswith("r8_"):
        print("Warning: Token 'r8_' ile baslamali")
    
    test_api(token)

