from django.shortcuts import render, get_object_or_404
from .models import Market, Results_Client
from django.contrib.auth.decorators import login_required
from datetime import datetime

@login_required
def client(request):
    markets = Market.objects.all()
    selected_market = None
    results = None

    if request.method == 'POST':
        market_name = request.POST.get('market')
        selected_market = get_object_or_404(Market, name=market_name)
        try:
            results = Results_Client.objects.filter(market=selected_market).latest("upload_at")
        except Results_Client.DoesNotExist:
            results = None

    return render(request, 'prediction/client_result.html', {
        'markets': markets,
        'selected_market': selected_market,
        'results': results,
    })
    
def generate_data_for_prediction(open_houre, clos_houre):
    time = datetime.now()
    if time < open_houre:
        return {"message": f"This market haven't opened yet. Please wait until {open_houre}"}
    elif time > clos_houre:
        return {"message": f"This market is closed now. Please try again later."}
    else:
        return 0
    
