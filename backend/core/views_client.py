from django.shortcuts import render, get_object_or_404
from .models import Market, Results_Client
from django.contrib.auth.decorators import login_required

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
