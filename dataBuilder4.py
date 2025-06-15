# Zachary Gruber
# Data Builder V4


import numpy as np
import requests
import pandas as pd
from datetime import datetime, timedelta
NUM_QUARTERS = 15
import time
from scipy.stats import zscore


CPI_DATA = pd.read_csv('CPILFESL.csv')
CPI_DATA = CPI_DATA.iloc[::-1]
CPI_DATA = CPI_DATA.reset_index(drop=True)

VIX_DATA = pd.read_csv('VIXCLS.csv')
VIX_DATA = VIX_DATA.iloc[::-1]
VIX_DATA = VIX_DATA.reset_index(drop=True)

current_time = datetime.now()

def isContiguous(table):
    contiguous = True

    for i in range(0,len(table)-1):
        date1 = table[i]
        date2 = table[i+1]
        difference = (date1-date2).n
        if difference != -1:
            contiguous = False

    return contiguous

def within_10_days(datetime1, datetime2):
    # Calculate the absolute difference between the two datetime objects
    time_difference = abs(datetime1 - datetime2)

    # Check if the absolute difference is less than or equal to 10 days
    if time_difference <= timedelta(days=10):
        return True
    else:
        return False

def remove_rows_with_outliers(df, column_name, threshold_factor=100):
    # Calculate median of the specified column
    median = df[column_name].median()

    # Calculate lower and upper thresholds
    lower_threshold = median / threshold_factor
    upper_threshold = median * threshold_factor

    # Filter rows based on the threshold
    cleaned_df = df[(df[column_name] >= lower_threshold) & (df[column_name] <= upper_threshold)]

    data = cleaned_df.reset_index(drop=True)

    return data

def remove_rows_with_outliers2(df, column_name, threshold):
    # Check if the column exists in the DataFrame
    if column_name not in df.columns:
        print(f"Column '{column_name}' not found in the DataFrame.")
        return df

    # Calculate the z-scores for the specified column
    df['ZScore'] = zscore(df[column_name])

    # Filter rows based on the z-score threshold condition
    filtered_df = df[abs(df['ZScore']) <= threshold]

    # Print a message with the number of removed rows
    removed_rows_count = len(df) - len(filtered_df)
    #print(f"Removed {removed_rows_count} rows outside the z-score threshold (±{threshold}).")

    # Drop the temporary 'ZScore' column before returning the filtered DataFrame
    filtered_df = filtered_df.drop(columns=['ZScore'])
    filtered_df = filtered_df.reset_index(drop=True)

    return filtered_df

def remove_rows_with_negative_values(df, column_name):
    # Filter rows based on the condition
    cleaned_df = df[df[column_name] >= 0]

    data = cleaned_df.reset_index(drop=True)

    return data

def remove_rows_with_no_shares(df, column_name):
    # Filter rows based on the condition
    cleaned_df = df[df[column_name] > 0]

    data = cleaned_df.reset_index(drop=True)

    return data

def remove_rows_with_zero(df, column_name):
    # Filter rows based on the condition
    cleaned_df = df[df[column_name] != 0]

    data = cleaned_df.reset_index(drop=True)

    return data

def getTotalAssets(index, sheet):
    dictionary_sheet = sheet['quarterlyReports'][index]
    value = int(dictionary_sheet['totalAssets'])
    return value

def getTotalAssetsDeltaY(index, sheet):
    assets_today = getTotalAssets(index, sheet)
    assets_lastquarter = getTotalAssets(index + 4, sheet)
    #change = ((assets_today - assets_lastquarter) / assets_lastquarter) * 100
    change = assets_today - assets_lastquarter
    return change

def getTotalLiabilities(index, sheet):
    dictionary_sheet = sheet['quarterlyReports'][index]
    value = dictionary_sheet['totalLiabilities']
    if value == "None":
        value = 100
    else:
        value = int(dictionary_sheet['totalLiabilities'])
        if value <= 0:
            value = 100
    return value

def getTotalLiabilitiesDeltaY(index, sheet):
    liab_today = getTotalLiabilities(index, sheet)
    liab_lastquarter = getTotalLiabilities(index + 4, sheet)
    #change = ((liab_today - liab_lastquarter) / liab_lastquarter) * 100
    change = liab_today - liab_lastquarter
    return change


def getIntanigbleAssets(index, sheet):
    dictionary_sheet = sheet['quarterlyReports'][index]
    value = dictionary_sheet['intangibleAssets']
    if value == "None":
        value = 100
    else:
        value = int(dictionary_sheet['intangibleAssets'])
        if value <= 0:
            value = 100
    return value

def getBookValue(index, sheet):
    assets = getTotalAssets(index, sheet)
    liabilites = getTotalLiabilities(index, sheet)
    intangible = getIntanigbleAssets(index, sheet)
    book_value = assets - intangible - liabilites
    return book_value

def getCash(index, sheet):
    dictionary_sheet = sheet['quarterlyReports'][index]
    value = dictionary_sheet['cashAndCashEquivalentsAtCarryingValue']
    if value == "None":
        value = 0
    else:
        value = int(dictionary_sheet['cashAndCashEquivalentsAtCarryingValue'])
        if value <= 0:
            value = 0
    return value

def getCashDelta(index, sheet):

    cash_today = (getCash(index, sheet) + getCash(index+1, sheet) + getCash(index+2, sheet) + getCash(index+3, sheet)) / 4
    cash_prev = (getCash(index+12, sheet) + getCash(index+13, sheet) + getCash(index+14, sheet) + getCash(index+15, sheet)) / 4

    if cash_prev == 0:
        if cash_today == 0:
            change = 0

        elif cash_today > 0:
            cash_prev = 10
            change = ((cash_today - cash_prev) / np.abs(cash_prev)) * 100

        else:
            cash_lastyr = -10
            change = ((cash_today - cash_lastyr) / np.abs(cash_lastyr)) * 100
    else:
        change = ((cash_today - cash_prev) / np.abs(cash_prev)) * 100

    return change

def getTreasuryStock(index, sheet):
    dictionary_sheet = sheet['quarterlyReports'][index]
    value = dictionary_sheet['treasuryStock']
    if value == "None":
        value = 0
    else:
        value = int(dictionary_sheet['treasuryStock'])
        if value <= 0:
            value = 0
    return value

def getTreasuryStockDelta(index, sheet):

    ts_today = getTreasuryStock(index, sheet)
    ts_prev = getTreasuryStock(index+12, sheet)

    if ts_prev == 0:
        if ts_today == 0:
            change = 0

        elif ts_today > 0:
            ts_prev = 10
            change = ((ts_today - ts_prev) / np.abs(ts_prev)) * 100

        else:
            ts_lastyr = -10
            change = ((ts_today - ts_lastyr) / np.abs(ts_lastyr)) * 100
    else:
        change = ((ts_today - ts_prev) / np.abs(ts_prev)) * 100

    return change

def getAvgCashPastYr(index, sheet):
    totalCash = getCash(index, sheet) + getCash(index+1, sheet) + getCash(index+2, sheet) + getCash(index+3, sheet)
    avgCash = totalCash / 4
    return avgCash

def getWeightedAvgCash(index, sheet):
    cash1 = getAvgCashPastYr(index, sheet) * .4
    cash2 = getAvgCashPastYr(index + 4, sheet) * .3
    cash3 = getAvgCashPastYr(index + 8, sheet) * .2
    cash4 = getAvgCashPastYr(index + 12, sheet) * .1
    totalCash = cash1 + cash2 + cash3 + cash4
    return totalCash

def getWorkingCapital(index, sheet):
    dictionary_sheet = sheet['quarterlyReports'][index]
    totalCurrentAssets = dictionary_sheet['totalCurrentAssets']
    totalCurrentLiabilities = dictionary_sheet['totalCurrentLiabilities']

    if totalCurrentAssets == "None" and totalCurrentLiabilities != "None":
        value = int(dictionary_sheet['totalCurrentLiabilities']) * -1
    elif totalCurrentLiabilities == "None" and totalCurrentAssets != "None":
        value = int(dictionary_sheet['totalCurrentAssets'])
    elif totalCurrentLiabilities == "None" and totalCurrentAssets == "None":
        value = 0
    else:
        value = int(dictionary_sheet['totalCurrentAssets']) - int(dictionary_sheet['totalCurrentLiabilities'])

    return value

def getWorkingCapitalDelta(index, sheet):

    wc_today = (getWorkingCapital(index, sheet) + getWorkingCapital(index+1, sheet) + getWorkingCapital(index+2, sheet) + getWorkingCapital(index+3, sheet)) / 4
    wc_lastyr = (getWorkingCapital(index+12, sheet) + getWorkingCapital(index+13, sheet) + getWorkingCapital(index+14, sheet) + getWorkingCapital(index+15, sheet)) / 4

    if wc_lastyr == 0:

        if wc_today == 0:
            change = 0

        elif wc_today > 0:
            wc_lastyr = 10
            change = ((wc_today - wc_lastyr) / np.abs(wc_lastyr)) * 100

        else:
            wc_lastyr = -10
            change = ((wc_today - wc_lastyr) / np.abs(wc_lastyr)) * 100
    else:
        change = ((wc_today - wc_lastyr) / np.abs(wc_lastyr)) * 100

    return change

def getOperatingIncome(index, sheet):
    dictionary_sheet = sheet['quarterlyReports'][index]
    value = dictionary_sheet['operatingIncome']
    if value == "None":
        value = 100
    else:
        value = int(dictionary_sheet['operatingIncome'])
        if value == 0:
            value = 100
    return value

def getOperatingIncomeDeltaQ(index, sheet):
    oi_today = getOperatingIncome(index, sheet)
    oi_lastquarter = getOperatingIncome(index+1, sheet)
    #change = ((oi_today - oi_lastquarter) / oi_lastquarter) * 100
    change = oi_today - oi_lastquarter
    return change

def getNetIncome(index, sheet):
    dictionary_sheet = sheet['quarterlyReports'][index]
    value = int(dictionary_sheet['netIncome'])
    return value

def getNetIncomeLastQ(index, sheet):
    value = getNetIncome(index + 1, sheet)
    return value

def getNetIncomeDeltaQ(index, sheet):
    ni_today = getNetIncome(index, sheet)
    ni_lastquarter = getNetIncome(index+1, sheet)
    #change = ((ni_today - ni_lastquarter) / ni_lastquarter) * 100
    change = ni_today - ni_lastquarter
    return change

def getNetIncomeDelta(index, sheet):

    ni_today = (getNetIncome(index, sheet) + getNetIncome(index+1, sheet) + getNetIncome(index+2, sheet) + getNetIncome(index+3, sheet)) / 4
    ni_lastyr= (getNetIncome(index+12, sheet) + getNetIncome(index+13, sheet) + getNetIncome(index+14, sheet) + getNetIncome(index+15, sheet)) / 4

    if ni_lastyr == 0:
        if ni_today == 0:
            change = 0

        elif ni_today > 0:
            ni_lastyr = 10
            change = ((ni_today - ni_lastyr) / np.abs(ni_lastyr)) * 100

        else:
            ni_lastyr = -10
            change = ((ni_today - ni_lastyr) / np.abs(ni_lastyr)) * 100

    else:
        change = ((ni_today - ni_lastyr) / np.abs(ni_lastyr)) * 100

    return change

def getTotalRevenue(index, sheet):
    dictionary_sheet = sheet['quarterlyReports'][index]
    value = dictionary_sheet['totalRevenue']
    if value == "None":
        value = 0
    else:
        value = int(dictionary_sheet['totalRevenue'])
    return value

def getNetIncomePastYr(index, sheet):
    tni_today = getNetIncome(index, sheet)
    tni_lastq = getNetIncome(index + 1, sheet)
    tni_lasttq = getNetIncome(index + 2, sheet)
    tni_lastttq = getNetIncome(index + 3, sheet)
    # change = ((tr_today - tr_lastquarter) / tr_lastquarter) * 100
    total = tni_today + tni_lastq + tni_lasttq + tni_lastttq
    return total

def getPreferredDividends(index, sheet1):
    dictionary_sheet = sheet1['quarterlyReports'][index]
    value = dictionary_sheet['dividendPayoutPreferredStock']
    if value == "None":
        value = 0
    else:
        value = int(dictionary_sheet['dividendPayoutPreferredStock'])
    return value

def getTrailingEPS(index, sheet1, sheet2, sheet3):
    avg_shares = (getShares(index, sheet1) + getShares(index+1, sheet1) + getShares(index+2, sheet1) + getShares(index+3, sheet1)) * 0.25
    totalIncome = getNetIncome(index, sheet2) + getNetIncome(index+1, sheet2) + getNetIncome(index+2, sheet2) + getNetIncome(index+3, sheet2)
    dividends = getPreferredDividends(index, sheet3) + getPreferredDividends(index+1, sheet3) + getPreferredDividends(index+2, sheet3) + getPreferredDividends(index+3, sheet3)
    eps = (totalIncome - dividends) / avg_shares
    return eps

def getTrailingEPS3Year(index, sheet1, sheet2, sheet3):
    eps1 = getTrailingEPS(index, sheet1, sheet2, sheet3)
    eps2 = getTrailingEPS(index+4, sheet1, sheet2, sheet3)
    eps3 = getTrailingEPS(index+8, sheet1, sheet2, sheet3)
    avg_eps = (eps1 + eps2 + eps3) / 3
    return avg_eps

def getTotalRevenueDeltaQ(index, sheet):
    tr_today = getTotalRevenue(index, sheet)
    tr_lastquarter = getTotalRevenue(index+1, sheet)
    #change = ((tr_today - tr_lastquarter) / tr_lastquarter) * 100
    change = tr_today - tr_lastquarter
    return change

def getTotalRevenueDelta(index, sheet):

    tr_today = (getTotalRevenue(index, sheet) + getTotalRevenue(index+1, sheet) + getTotalRevenue(index+2, sheet) + getTotalRevenue(index+3, sheet)) / 4
    tr_lastyr = (getTotalRevenue(index+12, sheet) + getTotalRevenue(index+13, sheet) + getTotalRevenue(index+14, sheet) + getTotalRevenue(index+15, sheet)) / 4

    if tr_lastyr < 0:
        return pd.NA

    elif tr_today < 0:
        return pd.NA

    elif tr_lastyr == 0:
        tr_lastyr = 10
        change = ((tr_today - tr_lastyr) / np.abs(tr_lastyr)) * 100

    else:
        change = ((tr_today - tr_lastyr) / np.abs(tr_lastyr)) * 100

    return change

def getRevenuePastYr(index, sheet):
    total_revenue = getTotalRevenue(index, sheet) + getTotalRevenue(index+1, sheet) + getTotalRevenue(index+2, sheet) + getTotalRevenue(index+3, sheet)
    return total_revenue

def getTotalRevenueDiffY(index, sheet):
    tr_today = getTotalRevenue(index, sheet)
    tr_lastyr = getTotalRevenue(index+4, sheet)
    change = tr_today - tr_lastyr
    return change

def getTotalRevenueLastQ(index, sheet):
    tr_lastquarter = getTotalRevenue(index+1, sheet)
    return tr_lastquarter

def getDividends(index, sheet):
    dictionary_sheet = sheet['quarterlyReports'][index]
    value = dictionary_sheet['dividendPayout']
    if value == "None":
        value = 100
    else:
        value = float(value)
        if value <= 0:
            value = 100
    return value

def getDividendsDeltaQ(index, sheet):
    d_today = getDividends(index, sheet)
    d_lastquarter = getDividends(index+1, sheet)
    #change = ((d_today - d_lastquarter) / d_lastquarter) * 100
    change = d_today - d_lastquarter
    return change

def getCashFlow(index,sheet1):
    dictionary_sheet = sheet1['quarterlyReports'][index]
    value1 = dictionary_sheet['operatingCashflow']
    if value1 == "None":
        value = 0
    else:
        value = int(dictionary_sheet['operatingCashflow'])
    return value

def getCashFlowDeltaQ(index,sheet1):
    cfd_today = getCashFlow(index, sheet1)
    cfd_lastquarter = getCashFlow(index + 1, sheet1)
    #change = ((cfd_today - cfd_lastquarter) / cfd_lastquarter) * 100
    change = cfd_today - cfd_lastquarter
    return change

def getCashFlowDelta(index,sheet1):
    cfd_today = (getCashFlow(index, sheet1) + getCashFlow(index+2, sheet1) + getCashFlow(index+3, sheet1) + getCashFlow(index+4, sheet1)) / 4
    cfd_lastyr = (getCashFlow(index + 12, sheet1) + getCashFlow(index+13, sheet1) + getCashFlow(index+14, sheet1) + getCashFlow(index+15, sheet1)) / 4
    if cfd_lastyr == 0:

        if cfd_today == 0:
            change = 0

        elif cfd_today > 0:
            cfd_lastyr = 10
            change = ((cfd_today - cfd_lastyr) / np.abs(cfd_lastyr)) * 100

        else:
            cfd_lastyr = -10
            change = ((cfd_today - cfd_lastyr) / np.abs(cfd_lastyr)) * 100

    else:
        change = ((cfd_today - cfd_lastyr) / np.abs(cfd_lastyr)) * 100

    return change

def getDebt(index, sheet1):
    dictionary_sheet = sheet1['quarterlyReports'][index]
    shortTermDebt = dictionary_sheet['shortTermDebt']
    longTermDebt = dictionary_sheet['longTermDebtNoncurrent']
    if shortTermDebt == "None" and longTermDebt != "None":
        shortTermDebt = 0
        longTermDebt = float(dictionary_sheet['longTermDebtNoncurrent'])
    elif shortTermDebt != "None" and longTermDebt == "None":
        shortTermDebt = float(dictionary_sheet['shortTermDebt'])
        longTermDebt = 0
    elif shortTermDebt != "None" and longTermDebt != "None":
        longTermDebt = float(dictionary_sheet['longTermDebtNoncurrent'])
        shortTermDebt = float(dictionary_sheet['shortTermDebt'])
        if longTermDebt + shortTermDebt == 0:
            longTermDebt = 0
            shortTermDebt = 0
    else:
        longTermDebt=0
        shortTermDebt=0
    value2 = longTermDebt + shortTermDebt
    return value2

def getDebt2(index, sheet):
    dictionary_sheet = sheet['quarterlyReports'][index]
    debt = dictionary_sheet['shortLongTermDebtTotal']
    if debt == "None":
        return 0
    else:
        debt = int(dictionary_sheet['shortLongTermDebtTotal'])
        return debt

def getAvgDebtPastYr(index, sheet):
    totalDebt = getDebt2(index, sheet) + getDebt2(index+1, sheet) + getDebt2(index+2, sheet) + getDebt2(index+3, sheet)
    avgDebt = totalDebt / 4
    return avgDebt

def getWeightedAvgDebt(index, sheet):
    debt1 = getAvgDebtPastYr(index, sheet) * .4
    debt2 = getAvgDebtPastYr(index + 4, sheet) * .3
    debt3 = getAvgDebtPastYr(index + 8, sheet) * .2
    debt4 = getAvgDebtPastYr(index + 12, sheet) * .1
    debt = debt1 + debt2 + debt3 + debt4
    return debt

def getDebtDelta(index, sheet1):

    debt_today = (getDebt(index, sheet1) + getDebt(index+1, sheet1) + getDebt(index+2, sheet1) + getDebt(index+3, sheet1)) / 4
    debt_lastyr = (getDebt(index + 12, sheet1) + getDebt(index+13, sheet1) + getDebt(index+14, sheet1) + getDebt(index+15, sheet1)) / 4

    if debt_today < 0 or debt_lastyr < 0:
        print('negative debt?')
        return pd.NA

    elif debt_lastyr == 0:

        if debt_today == 0:
            change = 0

        elif debt_today > 0:
            debt_lastyr = 10
            change = ((debt_today - debt_lastyr) / np.abs(debt_lastyr)) * 100

        else:
            debt_lastyr = -10
            change = ((debt_today - debt_lastyr) / np.abs(debt_lastyr)) * 100

    else:
        change = ((debt_today - debt_lastyr) / np.abs(debt_lastyr)) * 100

    return change

def getCashFlowtoDebt(index, sheet1, sheet2):
    dictionary_sheet = sheet1['quarterlyReports'][index]
    value1 = int(dictionary_sheet['operatingCashflow'])
    dictionary_sheet = sheet2['quarterlyReports'][index]
    shortTermDebt = dictionary_sheet['shortTermDebt']
    longTermDebt = dictionary_sheet['longTermDebtNoncurrent']
    if shortTermDebt == "None" and longTermDebt != "None":
        shortTermDebt = 0
        longTermDebt = int(dictionary_sheet['longTermDebtNoncurrent'])
    elif shortTermDebt != "None" and longTermDebt == "None":
        shortTermDebt = int(dictionary_sheet['shortTermDebt'])
        longTermDebt = 0
    elif shortTermDebt != "None" and longTermDebt != "None":
        longTermDebt = int(dictionary_sheet['longTermDebtNoncurrent'])
        shortTermDebt = int(dictionary_sheet['shortTermDebt'])
        if longTermDebt + shortTermDebt == 0:
            longTermDebt = 0
            shortTermDebt = 0
    else:
        longTermDebt=0
        shortTermDebt=0
    value2 = longTermDebt + shortTermDebt
    if value2 == 0:
        return pd.NA
    else:
        ratio = (value1/value2)
        return ratio

def getCFDDelta(index, sheet1, sheet2):
    cfd_today = getCashFlowtoDebt(index, sheet1, sheet2)
    cfd_lastyr = getCashFlowtoDebt(index+4, sheet1, sheet2)
    if pd.isna(cfd_today) or pd.isna(cfd_lastyr):
        return pd.NA

    elif cfd_lastyr == 0:
        return pd.NA

    else:
        change = ((cfd_today - cfd_lastyr) / cfd_lastyr) * 100
        return change

def getReturnOnAssets(index, sheet1, sheet2):
    netincome = getNetIncome(index, sheet1)
    averageAssets = (getTotalAssets(index, sheet2) + getTotalAssets(index+1, sheet2)) / 2
    ratio = netincome / averageAssets
    return ratio

def getReturnOnAssetsDeltaY(index, sheet1, sheet2):
    ROA_today = getReturnOnAssets(index, sheet1, sheet2)
    ROA_lastyear = getReturnOnAssets(index + 4, sheet1, sheet2)
    if ROA_lastyear == 0:
        return pd.NA
    else:
        change = ((ROA_today - ROA_lastyear) / ROA_lastyear) * 100
        return change

def getAssetTurnover(index, sheet1, sheet2):
    totalRevenue = getTotalRevenue(index, sheet1)
    averageAssets = (getTotalAssets(index, sheet2) + getTotalAssets(index+1, sheet2)) / 2
    ratio = totalRevenue / averageAssets
    return ratio

def getAssetTurnoverDeltaQ(index, sheet1, sheet2):
    AT_today = getAssetTurnover(index, sheet1, sheet2)
    AT_lastquarter = getAssetTurnover(index + 1, sheet1, sheet2)
    #change = ((AT_today - AT_lastquarter) / AT_lastquarter) * 100
    change = AT_today - AT_lastquarter
    return change

def getCOGS(index, sheet1):
    dictionary_sheet = sheet1['quarterlyReports'][index]
    value = dictionary_sheet['costofGoodsAndServicesSold']
    if value == 'None':
        value = 0
    else:
        value = int(dictionary_sheet['costofGoodsAndServicesSold'])
    return value

def getCOGSPastYr(index, sheet1):
    cogs = getCOGS(index, sheet1) + getCOGS(index+1, sheet1) + getCOGS(index+2, sheet1) + getCOGS(index+3, sheet1)
    return cogs

def getShares(index, sheet):
    dictionary_sheet = sheet['quarterlyReports'][index]
    shares = dictionary_sheet['commonStockSharesOutstanding']
    if shares != 'None':
        shares = int(dictionary_sheet['commonStockSharesOutstanding'])
    else:
        shares = 0

    return shares

def getEbitda(index, sheet):
    dictionary_sheet = sheet['quarterlyReports'][index]
    ebitda = dictionary_sheet['ebitda']
    if ebitda != 'None':
        ebitda = int(dictionary_sheet['ebitda'])
    else:
        ebitda = 0

    return ebitda

def getDividends2(index, sheet):
    dictionary_sheet = sheet['quarterlyReports'][index]
    dividends = dictionary_sheet['dividendPayout']
    if dividends != 'None':
        dividends = int(dictionary_sheet['dividendPayout'])
    else:
        dividends = 0

    return dividends

def getDividendRatio(index, sheet):
    ni = getNetIncome(index, sheet)
    dp = getDividends(index, sheet)
    ratio = dp/ni
    return ratio

def getEbitdaPastYr(index, sheet):
    totalEbitda = getEbitda(index, sheet) + getEbitda(index+1, sheet) + getEbitda(index+2, sheet) + getEbitda(index+3, sheet)
    return totalEbitda

def getTrailingEbitda(index, sheet):
    ebitda1 = getEbitdaPastYr(index, sheet) * .4
    ebitda2 = getEbitdaPastYr(index+4, sheet) * .3
    ebitda3 = getEbitdaPastYr(index+8, sheet) * .2
    ebitda4 = getEbitdaPastYr(index+12, sheet) * .1
    totalEbitda = ebitda1 + ebitda2 + ebitda3 + ebitda4
    return totalEbitda

def getEBITDADelta(index, sheet):
    ebitda_today = (getEbitda(index, sheet) + getEbitda(index+1, sheet) + getEbitda(index+2, sheet) + getEbitda(index+3, sheet)) / 4
    ebitda_lastY = (getEbitda(index + 12, sheet) + getEbitda(index+13, sheet) + getEbitda(index+14, sheet) + getEbitda(index+15, sheet)) / 4

    if ebitda_lastY == 0:
        if ebitda_today == 0:
            change = 0

        elif ebitda_today > 0:
            ebitda_lastyr = 10
            change = ((ebitda_today - ebitda_lastyr) / np.abs(ebitda_lastyr)) * 100

        else:
            ebitda_lastyr = -10
            change = ((ebitda_today - ebitda_lastyr) / np.abs(ebitda_lastyr)) * 100
    else:
        change = ((ebitda_today - ebitda_lastY) / np.abs(ebitda_lastY)) * 100

    return change

def getEBITDAtoSalesDelta(index, sheet):
    ebitda = (getEbitda(index, sheet) + getEbitda(index+1, sheet) + getEbitda(index+2, sheet) + getEbitda(index+3, sheet)) / 4
    sales = (getTotalRevenue(index, sheet) + getTotalRevenue(index+1, sheet) + getTotalRevenue(index+2, sheet) + getTotalRevenue(index+3, sheet)) / 4

    if sales == 0:
        return pd.NA
    else:
        ratio_new = ebitda / sales

    ebitda2 = (getEbitda(index+12, sheet) + getEbitda(index + 13, sheet) + getEbitda(index + 14, sheet) + getEbitda(index + 15,sheet)) / 4
    sales2 = (getTotalRevenue(index+12, sheet) + getTotalRevenue(index + 13, sheet) + getTotalRevenue(index+14,sheet) + getTotalRevenue(index+15, sheet)) / 4


    if sales2 == 0:
        return pd.NA
    else:
        ratio_prev = ebitda2/sales2

    if ratio_prev == 0:
        if ratio_new == 0:
            change = 0

        elif ratio_new > 0:
            ratio_prev = 0.0001
            change = ((ratio_new - ratio_prev) / np.abs(ratio_prev)) * 100

        else:
            ratio_prev = -0.0001
            change = ((ratio_new - ratio_prev) / np.abs(ratio_prev)) * 100
    else:
        change = ((ratio_new - ratio_prev) / np.abs(ratio_prev)) * 100

    return change

def getOItoSalesDelta(index, sheet):
    opinc = (getOperatingIncome(index, sheet) + getOperatingIncome(index+1, sheet) + getOperatingIncome(index+2, sheet) + getOperatingIncome(index+3, sheet)) / 4
    sales = (getTotalRevenue(index, sheet) + getTotalRevenue(index+1, sheet) + getTotalRevenue(index+2, sheet) + getTotalRevenue(index+3, sheet)) / 4
    if sales == 0:
        return pd.NA
    else:
        ratio_new = opinc / sales

    opinc2 = (getOperatingIncome(index + 12, sheet) + getOperatingIncome(index+13, sheet) + getOperatingIncome(index+14, sheet) + getOperatingIncome(index+15, sheet)) / 4
    sales2 = (getTotalRevenue(index + 12, sheet) + getTotalRevenue(index+13, sheet) + getTotalRevenue(index+14, sheet) + getTotalRevenue(index+15, sheet)) / 4

    if sales2 == 0:
        return pd.NA
    else:
        ratio_prev = opinc2 / sales2

    if ratio_prev == 0:
        if ratio_new == 0:
            change = 0

        elif ratio_new > 0:
            ratio_prev = 0.0001
            change = ((ratio_new - ratio_prev) / np.abs(ratio_prev)) * 100

        else:
            ratio_prev = -0.0001
            change = ((ratio_new - ratio_prev) / np.abs(ratio_prev)) * 100
    else:
        change = ((ratio_new - ratio_prev) / np.abs(ratio_prev)) * 100

    return change

def getSharesDeltaQ(index, sheet):
    shares_today = getShares(index, sheet)
    shares_lastQ = getShares(index+1, sheet)
    change = ((shares_today - shares_lastQ) / shares_lastQ) * 100
    return change

def getDate(index, sheet3):
    dictionary_sheet = sheet3['quarterlyReports'][index]
    date = dictionary_sheet['fiscalDateEnding']
    return date

def balanceSheetCall(ticker):
    r = requests.get(f'https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={ticker}&apikey={KEY}')
    data = r.json()
    return data

def incomeSheetCall(ticker):
    r = requests.get(f'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={ticker}&apikey={KEY}')
    data = r.json()
    return data

def cashFlowSheetCall(ticker):
    r = requests.get(f'https://www.alphavantage.co/query?function=CASH_FLOW&symbol={ticker}&apikey={KEY}')
    data = r.json()
    return data

def earningsCall(ticker):
    r = requests.get(f'https://www.alphavantage.co/query?function=EARNINGS&symbol={ticker}&apikey={KEY}')
    data = r.json()
    return data

def stockPricesCall(ticker):
    r = requests.get(f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&outputsize=full&apikey={KEY}')
    data = r.json()
    return data

def wmaCall(ticker):
    r = requests.get(f'https://www.alphavantage.co/query?function=WMA&symbol={ticker}&interval=daily&time_period=10&series_type=open&apikey={KEY}')
    data = r.json()
    return data

def dmaCall(ticker):
    r = requests.get(f'https://www.alphavantage.co/query?function=DEMA&symbol={ticker}&interval=daily&time_period=10&series_type=close&apikey={KEY}')
    data = r.json()
    return data

def mamafamaCall(ticker):
    r = requests.get(f'https://www.alphavantage.co/query?function=MAMA&symbol={ticker}&interval=daily&series_type=close&fastlimit=0.02&apikey={KEY}')
    data = r.json()
    return data

def adxCall(ticker):
    r = requests.get(f'https://www.alphavantage.co/query?function=ADX&symbol={ticker}&interval=daily&time_period=10&apikey={KEY}')
    data = r.json()
    return data

def rsiCall(ticker):
    r = requests.get(f'https://www.alphavantage.co/query?function=RSI&symbol={ticker}&interval=daily&time_period=10&series_type=close&apikey={KEY}')
    data = r.json()
    return data

def stochRsiCall(ticker):
    r = requests.get(f'https://www.alphavantage.co/query?function=STOCHRSI&symbol={ticker}&interval=daily&time_period=10&series_type=open&fastkperiod=10&fastdmatype=0&apikey={KEY}')
    data = r.json()
    return data

def mfiCall(ticker):
    r = requests.get(f'https://www.alphavantage.co/query?function=MFI&symbol={ticker}&interval=daily&time_period=10&apikey={KEY}')
    data = r.json()
    return data

def apoCall(ticker):
    r = requests.get(f'https://www.alphavantage.co/query?function=APO&symbol={ticker}&interval=daily&series_type=close&fastperiod=10&slowperiod=20&matype=8&apikey={KEY}')
    data = r.json()
    return data

def statistics_fixed(ticker,date1, date2):
    r = requests.get(f'https://alphavantageapi.co/timeseries/analytics?SYMBOLS={ticker},SPY&RANGE={date1}&RANGE={date2}&INTERVAL=DAILY&OHLC=close&CALCULATIONS=VARIANCE(annualized=True),COVARIANCE(annualized=True),STDDEV(annualized=True)&apikey={KEY}')
    data = r.json()
    return data

def statistics_sliding(ticker, date1, date2, window):
    r = requests.get(f'https://www.alphavantage.co/query?function=ANALYTICS_SLIDING_WINDOW&SYMBOLS={ticker},SPY&RANGE={date1}&RANGE={date2}&INTERVAL=DAILY&OHLC=open&WINDOW_SIZE={window}&CALCULATIONS=VARIANCE(annualized=True),COVARIANCE(annualized=True),STDDEV(annualized=True),CORRELATION&apikey={KEY}')
    #print(f'https://www.alphavantage.co/query?function=ANALYTICS_SLIDING_WINDOW&SYMBOLS={ticker},SPY&RANGE={date1}&RANGE={date2}&INTERVAL=WEEKLY&OHLC=open&WINDOW_SIZE={window}&CALCULATIONS=VARIANCE(annualized=True),COVARIANCE(annualized=True),STDDEV(annualized=True),CORRELATION&apikey={KEY}')
    data = r.json()
    return data

def getTyield():
    r = requests.get(f'https://www.alphavantage.co/query?function=TREASURY_YIELD&interval=daily&maturity=10year&apikey={KEY}')
    data = r.json()
    return data

def getRGDP():
    r = requests.get(f'https://www.alphavantage.co/query?function=REAL_GDP&interval=quarterly&apikey={KEY}')
    data = r.json()
    return data

# def getCPI():
#     r = requests.get(f'https://www.alphavantage.co/query?function=CPI&interval=monthly&apikey={KEY}')
#     data = r.json()
#     return data

def createFinancialRows(ticker, index, bal_sheet, inc_sheet, cash_sheet):

    #fetch data
    BALANCE_SHEET = bal_sheet
    INCOME_SHEET = inc_sheet
    CASHFLOW_SHEET = cash_sheet

    #create list
    lst = []

    #features
    lst.append(getCashDelta(index, BALANCE_SHEET))
    lst.append(getWorkingCapitalDelta(index, BALANCE_SHEET))
    lst.append(getNetIncomeDelta(index, INCOME_SHEET))
    lst.append(getTotalRevenueDelta(index, INCOME_SHEET))
    lst.append(getCashFlowDelta(index, CASHFLOW_SHEET))
    lst.append(getEBITDADelta(index, INCOME_SHEET))
    lst.append(getDebtDelta(index, BALANCE_SHEET))
    lst.append(getEBITDAtoSalesDelta(index, INCOME_SHEET))
    lst.append(getOItoSalesDelta(index, INCOME_SHEET))
    lst.append(getTreasuryStockDelta(index, BALANCE_SHEET))

    #dividends are diff
    lst.append(getDividends2(index, CASHFLOW_SHEET))

    #needed for E.V.
    lst.append(getAvgDebtPastYr(index, BALANCE_SHEET))
    lst.append(getAvgCashPastYr(index, BALANCE_SHEET))

    #shares
    lst.append(getShares(index, BALANCE_SHEET))

    #other data
    lst.append(ticker)
    lst.append(getDate(index, BALANCE_SHEET))

    return lst

def createDateRows(index, data):

    lst = []
    dictionary_sheet = data['quarterlyEarnings'][index]
    fiscalDateEnding = dictionary_sheet['fiscalDateEnding']

    reportDate = dictionary_sheet['reportedDate']
    if dictionary_sheet['reportedEPS'] == "None":
        reportEPS = pd.NA
    else:
        reportEPS = float(dictionary_sheet['reportedEPS'])

    surprisePercentage = dictionary_sheet['surprisePercentage']
    if surprisePercentage == "None":
        surprisePercentage = pd.NA
    else:
        surprisePercentage = float(dictionary_sheet['surprisePercentage'])
    lst.append(fiscalDateEnding)
    lst.append(reportDate)
    lst.append(reportEPS)
    lst.append(surprisePercentage)
    return lst

def createOffsetRows(table):
    offset = 1
    lst = []
    for date in table['reportedDate']:
        date = str(date)
        date_formatted = date[:10]

        #get adjusted date if needed
        #negative moves to future date
        offset_date = beginDate(-1, date_formatted)

        lst.append([offset_date])

    return lst

def createStockRows(ticker, table):
    lst = []
    stock = stockPricesCall(ticker)
    data = stock['Time Series (Daily)']
    split_coefficient = 1
    dividend_amount = 0

    for date in table['offset_date']:

        try:
            before,after = beforeAndAfter(date)

        except ValueError:
            print('inaccurate report date')
            lst.append([pd.NA, pd.NA, pd.NA])

        else:
            #prev_close = float(data[before]['4. close'])
            next_open = float(data[after]['1. open'])
            next_adjusted_close = float(data[after]['5. adjusted close'])

            #lst.append([next_open, prev_close, next_adjusted_close])
            lst.append([next_open, next_adjusted_close])

    # for i in range(len(keys_list)):
    #     date = keys_list[i]
    #
    #     #keep a running count of dividends and splits
    #     split_coefficient *= float(data[date]['8. split coefficient'])
    #     open = float(data[date]['1. open'])
    #
    #     if i < len(keys_list) - 1:
    #         match_date = keys_list[i + 1]
    #         prev_match_date = keys_list[i + 2]
    #
    #         # Parse the string into a datetime object
    #         match_date_strp = datetime.strptime(match_date, '%Y-%m-%d')
    #         if match_date_strp == table.iloc[index]['reportedDate']:
    #             prev_close = float(data[prev_match_date]['4. close'])
    #             change_price = ((open - prev_close) / np.abs(prev_close)) * 100
    #             lst.append([open, prev_close, split_coefficient, change_price])
    #             index += 1
    #             if index == table.shape[0]:
    #                 break

    return lst

def techPrices(ticker, table):
    lst = []

    indicator1 = mamafamaCall(ticker)
    mama_fama_data = indicator1['Technical Analysis: MAMA']

    indicator2 = adxCall(ticker)
    adx_data = indicator2['Technical Analysis: ADX']

    indicator3 = rsiCall(ticker)
    rsi_data = indicator3['Technical Analysis: RSI']

    indicator4 = stochRsiCall(ticker)
    stochRsi_data = indicator4['Technical Analysis: STOCHRSI']

    indicator5 = mfiCall(ticker)
    mfi_data = indicator5['Technical Analysis: MFI']

    indicator6 = apoCall(ticker)
    apo_data = indicator6['Technical Analysis: APO']

    for date in table['reportedDate']:
        #figure out date
        day_before = beginDate(1,date.strftime("%Y-%m-%d"))

        #indicator 1 - MAMA FAMA
        mama = float(mama_fama_data[day_before]['MAMA'])
        fama = float(mama_fama_data[day_before]['FAMA'])

        if mama > fama:
            upward = 1
        else:
            upward = 0

        #indicator 2 - ADX
        adx = float(adx_data[day_before]['ADX'])
        if adx > 30:
            strong = 1
        else:
            strong = 0

        #indicator 3 - RSI one-hot encoded
        rsi = float(rsi_data[day_before]['RSI'])
        if rsi > 75:
            rsi_overbought = 1
        else:
            rsi_overbought = 0

        if rsi < 25:
            rsi_oversold = 1
        else:
            rsi_oversold = 0

        # indicator 4 - Stoch RSI one-hot encoded
        stoch_rsi = float(stochRsi_data[day_before]['FastK'])
        if stoch_rsi > 80:
            stoch_rsi_overbought = 1
        else:
            stoch_rsi_overbought = 0
        if stoch_rsi < 20:
            stoch_rsi_oversold = 1
        else:
            stoch_rsi_oversold = 0

        # indicator 5 - mfi one-hot encoded
        mfi = float(mfi_data[day_before]['MFI'])
        if mfi > 80:
            mfi_overbought = 1
        else:
            mfi_overbought = 0

        if mfi < 20:
            mfi_oversold = 1
        else:
            mfi_oversold = 0

        # indicator 6 - apo
        apo = float(apo_data[day_before]['APO'])
        if apo > 0:
            apo_bullish = 1
        else:
            apo_bullish = 0

        #append the indicators
        lst.append([upward, strong, rsi_overbought, rsi_oversold, stoch_rsi_overbought, stoch_rsi_oversold, mfi_overbought, mfi_oversold, apo_bullish])

    return lst

# def createSplitRows(ticker, table):
#     stock = stockPricesCall(ticker)
#     data = stock['Time Series (Daily)']
#     last_date =

def createFixedRows(table, ticker):

    lst = []
    dates = table['reportedDate']

    for i in range(len(dates)):

        #date calculations
        date_today = dates.iloc[i].strftime("%Y-%m-%d")
        #date_prev = dates.iloc[i+1].strftime("%Y-%m-%d")
        date_prev = beginDate(100,date_today)

        data = statistics_fixed(ticker, date_prev, date_today)
        time.sleep(0.2)
        data = data['payload']['RETURNS_CALCULATIONS']
        index_variance = data['VARIANCE(ANNUALIZED=TRUE)']['SPY']
        covariance = data['COVARIANCE(ANNUALIZED=TRUE)']['covariance'][1][0]
        if index_variance != 0:
            beta = covariance / index_variance
        else:
            beta = pd.NA
        vol = data['STDDEV(ANNUALIZED=TRUE)'][ticker]
        lst.append([beta,vol])

    return lst

def createSlidingRows(table, ticker):
    lst = []
    dates = table['reportedDate']
    window_size = 200

    for i in range(len(dates) - 1):
        # date calculations
        date_today = dates.iloc[i].strftime("%Y-%m-%d")
        # date_prev = dates.iloc[i+1].strftime("%Y-%m-%d")
        date_prev = beginDate(window_size, date_today)

        data = statistics_sliding(ticker, date_prev, date_today, window_size)['payload']['RETURNS_CALCULATIONS']

        # #get covariance
        # dictionary = data['COVARIANCE(ANNUALIZED=TRUE)']['RUNNING_COVARIANCE']
        # key_list = list(dictionary.keys())
        # middle_key = key_list[1]
        # dictionary_2 = dictionary[middle_key]
        # key_list2 = list(dictionary_2.keys())
        # key = key_list2[0]
        # covariance = dictionary_2[key]
        #
        # #get index variance
        # dictionary_3 = data['VARIANCE(ANNUALIZED=TRUE)']['RUNNING_VARIANCE']['SPY']
        # key_list = list(dictionary_3.keys())
        # key = key_list[0]
        # index_variance = dictionary_3[key]
        #
        # # calculate beta
        # beta = index_variance / covariance

        #calculate volatility (ticker std)
        dictionary_4 = data['STDDEV(ANNUALIZED=TRUE)']['RUNNING_STDDEV'][ticker]
        key_list = list(dictionary_4.keys())
        key = key_list[0]
        vol = dictionary_4[key]
        vol = vol*np.sqrt(252)*np.sqrt(252)

        # #get correlation
        # dictionary_5 = data['CORRELATION']['RUNNING_CORRELATION']
        # key_list = list(dictionary_5.keys())
        # key = key_list[0]
        # dictionary_6 = dictionary_5[key]
        # key_list = list(dictionary_6.keys())
        # key = key_list[0]
        # r = dictionary_6[key]

        #lst.append([beta, vol, r])
        lst.append([vol])

    return lst

def beginDate(days, start_date):
    stock = stockPricesCall('AAPL')
    dict = stock['Time Series (Daily)']
    lst = list(dict.keys())
    index = lst.index(start_date)
    index += days
    end_date = lst[index]
    return end_date

def beforeAndAfter(date):
    stock = stockPricesCall('AAPL')
    dict = stock['Time Series (Daily)']
    lst = list(dict.keys())
    index = lst.index(date)
    date1 = index + 1
    date2 = index - 1
    end_date = lst[date1]
    start_date = lst[date2]

    return end_date,start_date

def create10YRYieldsRows(table):
    lst = []
    index = 0
    rates = getTyield()
    data = rates['data']
    treasury = 4.09
    for i in range(len(data)):
        dict = data[i]
        if dict['value'] != ".":
            treasury = float(dict['value'])

        if i < len(data) - 1:
            next_dict = data[i+1]
            next_date = next_dict['date']
            # Parse the string into a datetime object
            # next_date = datetime.strptime(next_date, '%Y-%m-%d')
            if next_date == table.iloc[index]['offset_date']:
                lst.append(treasury)
                index+=1
                if index == table.shape[0]:
                    break
    return lst

def createRGDPRows(table):
    lst = []
    index = 0
    main_dictionary = getRGDP()
    data = main_dictionary['data']
    for i in range(len(data)):
        dict = data[i]
        if dict['value'] != ".":
            gdp = float(dict['value'])
            date = dict['date']
            date = datetime.strptime(date, '%Y-%m-%d')

            if date < table.iloc[index]['reportedDate']:
                dict2 = data[i + 4]
                gdp2 = float(dict2['value'])
                change = ((gdp-gdp2) / gdp2) * 100
                lst.append(change)
                index+=1
                if index == table.shape[0]:
                    break
    return lst

def createCPIRows(table):
    lst = []
    index = 0
    for i in range(len(CPI_DATA['DATE'])):
        cpi1 = CPI_DATA.iloc[i, 1]
        cpi2 = CPI_DATA.iloc[i + 12, 1]
        change = ((cpi1 - cpi2) / cpi2) * 100
        date = CPI_DATA.iloc[i,0]
        date = datetime.strptime(date, '%Y-%m-%d')
        if date < table.iloc[index]['reportedDate']:
            lst.append(change)
            index+=1
            if index == table.shape[0]:
                break
    return lst

def createVIXRows(table):
    lst = []
    index = 0
    for i in range(len(VIX_DATA['DATE'])):
        vix = VIX_DATA.iloc[i,1]
        date = VIX_DATA.iloc[i,0]
        date = datetime.strptime(date, '%Y-%m-%d')
        if date < table.iloc[index]['reportedDate']:
            if vix == '.':
                vix = VIX_DATA.iloc[i+1, 1]
            lst.append(vix)
            index+=1
            if index == table.shape[0]:
                break
    return lst

def enumerateTime(df):
    stock = stockPricesCall('AAPL')
    dict = stock['Time Series (Daily)']
    lst = list(dict.keys())
    lst2 = []
    base_date = lst.index('2012-01-03')

    #chance here you get a weekend date?
    for date in df['offset_date']:
        #if using reported date
        #date_string = date.strftime("%Y-%m-%d %H:%M:%S")
        #query = lst.index(date_string[:10])
        query = lst.index(date)
        difference = base_date - query
        lst2.append(difference)

    return lst2



















#main function to create a dataframe from a ticker
def createDataFrame(ticker, time):
    table = []
    BALANCE_SHEET = balanceSheetCall(ticker)
    INCOME_SHEET = incomeSheetCall(ticker)
    CASHFLOW_SHEET = cashFlowSheetCall(ticker)
    dates = earningsCall(ticker)

    #make sure company has enough reports
    quarterly_report_count = len(BALANCE_SHEET['quarterlyReports'])
    quarterly_earnings_count = len(dates['quarterlyEarnings'])

    max_quarters = min(quarterly_earnings_count, quarterly_report_count) - 5

    if max_quarters < time:
        time = max_quarters
        print("Adjusting quarters for " + ticker + " to " + str(time))

    for i in range(time):
        row = createFinancialRows(ticker, i, BALANCE_SHEET, INCOME_SHEET, CASHFLOW_SHEET)
        table.append(row)


    df = pd.DataFrame(table)
    df.columns = [

        #main features
        'cashDelta',
        'workingCapitalDelta',
        'netIncomeDelta',
        'totalRevenueDelta',
        'cashFlowDelta',
        'EBITDADelta',
        'debtDelta',
        'EBITDAtoSalesDelta',
        'OpIncometoSalesDelta',
        'treasuryStockDelta',
        'dividends',

        #for EV
        'avgCash',
        'avgDebt',

        #other data
        'sharesOutstanding',
        'ticker',
        'fiscalDateEnding1'

    ]

    df = df.dropna().reset_index(drop=True)

    table2 = []
    for i in range(time):
        row = createDateRows(i, dates)
        table2.append(row)

    df2 = pd.DataFrame(table2)

    df2.columns = [
        'fiscalDateEnding2',
        'reportedDate',
        'reportedEPS',
        'surprisePercentage'
    ]

    #run test
    date1 = df.iloc[0]['fiscalDateEnding1']
    date2 = df2.iloc[0]['fiscalDateEnding2']
    date1 = datetime.strptime(date1, '%Y-%m-%d')
    date2 = datetime.strptime(date2, '%Y-%m-%d')
    if within_10_days(date1, date2):
        merged_table = pd.concat([df, df2], axis=1)
        merged_table['reportedDate'] = pd.to_datetime(merged_table['reportedDate'])

    #AlphaVantage didn't add earnings yet
    else:
        table2 = []
        for i in range(1,time+1):
            test = i
            row = createDateRows(i, dates)
            table2.append(row)

        df2 = pd.DataFrame(table2)

        df2.columns = [
            'fiscalDateEnding2',
            'reportedDate',
            'reportedEPS',
            'surprisePercentage'
        ]
        merged_table = pd.concat([df, df2], axis=1)
        merged_table['reportedDate'] = pd.to_datetime(merged_table['reportedDate'])


    #check if the first report date was less than a few days ago
    first_report = merged_table.loc[0,'reportedDate']
    three_days_ago = current_time - timedelta(days=3)
    if first_report >= pd.Timestamp(three_days_ago):
        merged_table = merged_table.drop(merged_table.index[0]).reset_index(drop=True)

    #create offset on table
    data = createOffsetRows(merged_table)
    offset_table = pd.DataFrame(data)
    offset_table.columns = [
        'offset_date'
    ]
    merged_table = pd.concat([merged_table, offset_table], axis=1)
    merged_table = merged_table.dropna().reset_index(drop=True)


    #add stock prices next open day
    data = createStockRows(ticker,merged_table)
    stock_table = pd.DataFrame(data)
    stock_table.columns = [
        'next_unadjusted_open',
        'next_adjusted_close'
    ]
    merged_table = pd.concat([merged_table, stock_table], axis=1)
    merged_table = merged_table.dropna().reset_index(drop=True)

    #add WMA
    # data = techPrices(ticker, merged_table)
    # techIndicators_table = pd.DataFrame(data)
    # techIndicators_table.columns = [
    #     'trend_direction_mamafama',
    #     'trend_strength_adx',
    #     'overbought_rsi',
    #     'oversold_rsi',
    #     'overbought_stochrsi',
    #     'oversold_stochrsi',
    #     'overbought_mfi',
    #     'oversold_mfi',
    #     'trend_direction_apo'
    # ]
    # merged_table = pd.concat([merged_table, techIndicators_table], axis=1)
    # merged_table = merged_table.dropna().reset_index(drop=True)


    # #stock split adjustments
    # data = createSplitRows(merged_table,ticker)
    # split = pd.DataFrame(data)
    # split.columns = [
    #     'split_factor'
    # ]
    # merged_table = pd.concat([merged_table, split], axis=1)
    # merged_table = merged_table.dropna().reset_index(drop=True)

    # #add fixed analytics rows
    # data = createFixedRows(merged_table,ticker)
    # fixed = pd.DataFrame(data)
    # fixed.columns = [
    #     'beta',
    #     'AnnualizedVolatility'
    # ]
    # merged_table = pd.concat([merged_table, fixed], axis=1)

    #add sliding analytics rows
    data = createSlidingRows(merged_table,ticker)
    sliding = pd.DataFrame(data)
    sliding.columns = [
        'AnnualizedVolatility',
    ]
    merged_table = pd.concat([merged_table, sliding], axis=1)

    # #add 10YR rows
    # data = create10YRYieldsRows(merged_table)
    # if(len(data) == 0):
    #     merged_table = merged_table.drop(merged_table.index[0])
    #     merged_table = merged_table.reset_index(drop=True)
    #     data = create10YRYieldsRows(merged_table)
    # yields = pd.DataFrame(data)
    # yields.columns = [
    #     '10YR'
    # ]
    # merged_table = pd.concat([merged_table, yields], axis=1)
    # merged_table = merged_table.dropna().reset_index(drop=True)
    #
    # data = createRGDPRows(merged_table)
    # if (len(data) == 0):
    #     merged_table = merged_table.drop(merged_table.index[0])
    #     merged_table = merged_table.reset_index(drop=True)
    #     data = createRGDPRows(merged_table)
    # rgdp = pd.DataFrame(data)
    # rgdp.columns = [
    #     'GDPDeltaY'
    # ]
    # merged_table = pd.concat([merged_table, rgdp], axis=1)
    # merged_table = merged_table.dropna().reset_index(drop=True)
    #
    # data = createCPIRows(merged_table)
    # if (len(data) == 0):
    #     merged_table = merged_table.drop(merged_table.index[0])
    #     merged_table = merged_table.reset_index(drop=True)
    #     data = createCPIRows(merged_table)
    # cpi = pd.DataFrame(data)
    # cpi.columns = [
    #     'CPIDeltaY'
    # ]
    # merged_table = pd.concat([merged_table, cpi], axis=1)
    # merged_table = merged_table.dropna().reset_index(drop=True)

    # data = createVIXRows(merged_table)
    # if (len(data) == 0):
    #     merged_table = merged_table.drop(merged_table.index[0])
    #     merged_table = merged_table.reset_index(drop=True)
    #     data = createVIXRows(merged_table)
    # vix = pd.DataFrame(data)
    # vix.columns = [
    #     'VIX'
    # ]
    # merged_table = pd.concat([merged_table, vix], axis=1)
    # merged_table = merged_table.dropna().reset_index(drop=True)

    data = enumerateTime(merged_table)
    days = pd.DataFrame(data)
    days.columns = [
         'day'
    ]
    merged_table = pd.concat([merged_table, days], axis=1)
    merged_table = merged_table.dropna().reset_index(drop=True)

    #Clean and format data
    merged_table = remove_rows_with_no_shares(merged_table, 'sharesOutstanding')
    merged_table = remove_rows_with_outliers(merged_table,'sharesOutstanding',threshold_factor=100)
    #merged_table = remove_rows_with_negative_values(merged_table,'debt')
    merged_table['marketCap'] = merged_table['sharesOutstanding'] * merged_table['next_unadjusted_open']
    merged_table['dividendYield'] = (merged_table['dividends'] / merged_table['sharesOutstanding']) / merged_table['next_unadjusted_open']
    #merged_table['evNew'] = merged_table['marketCapNew'] + merged_table['avgDebt'] - merged_table['avgCash']

    #clean date rows
    rows_to_delete = []
    for index, row in merged_table.iterrows():
        date1 = merged_table.iloc[index]['fiscalDateEnding1']
        date2 = merged_table.iloc[index]['fiscalDateEnding2']
        date1 = datetime.strptime(date1, '%Y-%m-%d')
        date2 = datetime.strptime(date2, '%Y-%m-%d')
        if not within_10_days(date1, date2):
            rows_to_delete.append(index)

    merged_table.drop(rows_to_delete,inplace=True)
    merged_table.reset_index(drop=True, inplace=True)

    # # generate scores
    # merged_table['evOld'] = pd.NA
    merged_table['adjusted_close_change'] = pd.NA
    merged_table['mc_change'] = pd.NA
    merged_table.loc[0, 'adjusted_close_change'] = ((merged_table.loc[0, 'next_adjusted_close'] - merged_table.loc[11, 'next_adjusted_close']) / np.abs(merged_table.loc[11, 'next_adjusted_close'])) * 100
    merged_table.loc[0, 'mc_change'] = ((merged_table.loc[0, 'marketCap'] - merged_table.loc[11, 'marketCap']) / np.abs(merged_table.loc[11, 'marketCap'])) * 100
    avgDivPayout = (merged_table['dividendYield'].sum()/3) * 100
    merged_table.loc[0, 'avgDividendYield'] = avgDivPayout

    merged_table = merged_table.dropna().reset_index(drop=True)

    merged_table['volAdjusted_close_change'] = merged_table['adjusted_close_change'] / merged_table['AnnualizedVolatility']
    merged_table['volAdjusted_mc_change'] = merged_table['mc_change'] / merged_table['AnnualizedVolatility']

    return merged_table


# #Create training data
# file_path = 'SP500.csv'
# df = pd.read_csv(file_path)
# stock_list = df['Symbol'].values.tolist()
# dataset = pd.DataFrame()
#
# for i in range(len(stock_list)):
#     try:
#         print('Adding: ' + stock_list[i])
#         stock_data = createDataFrame(stock_list[i],NUM_QUARTERS)
#         dataset = pd.concat([dataset, stock_data], ignore_index=True)
#
#     except Exception as e:
#         print('Error thrown by ' + stock_list[i] + str(e))
#         continue
#
# dataset.to_csv('datasetSP.csv',index=False)




# #Create test data
# production_list = ["VIRT", "KKR"]
# set2 = set(production_list)
# dataset2 = pd.DataFrame()
#
# for i in range(len(production_list)):
#     try:
#         print('Adding: ' + production_list[i])
#         stock_data2 = createDataFrame(production_list[i], NUM_QUARTERS)
#         dataset2 = pd.concat([dataset2, stock_data2], ignore_index=True)
#
#     except Exception as e:
#         print('Error thrown by ' + production_list[i] + str(e))
#         continue
#
# dataset2.to_csv('datasetTest.csv',index=False)
