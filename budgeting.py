#!/usr/bin/env python3
"""
Budgeting module: Shows when expenses will occur based on order_by_date from cannabis logistics.
Groups orders into time buckets and shows total cost per period.
"""
from datetime import datetime, timezone
from typing import Dict, List, Optional

from cannabis_logistics import get_all_forecasts, parse_iso8601_z


def group_by_time_buckets(forecasts: List[Dict], as_of: Optional[datetime] = None) -> Dict[str, List[Dict]]:
	"""Group forecasts by time buckets based on order_by_date.
	
	Buckets:
	- "this_week": next 7 days
	- "next_2_weeks": days 8-14
	- "this_month": days 15-30
	- "beyond": > 30 days
	
	Args:
		forecasts: List of forecast dictionaries
		as_of: Reference time (defaults to now)
	
	Returns:
		Dictionary mapping bucket names to lists of forecasts
	"""
	if as_of is None:
		as_of = datetime.now(timezone.utc)
	
	buckets = {
		"this_week": [],
		"next_2_weeks": [],
		"this_month": [],
		"beyond": []
	}
	
	for f in forecasts:
		# Skip finished packages, reordered packages, or packages without order_by_date
		if f.get("finished") or f.get("reordered") or not f.get("order_by_date") or not f.get("package_cost"):
			continue
		
		try:
			order_date = parse_iso8601_z(f["order_by_date"])
			days_until = (order_date - as_of).days
			
			if days_until <= 7:
				buckets["this_week"].append(f)
			elif days_until <= 14:
				buckets["next_2_weeks"].append(f)
			elif days_until <= 30:
				buckets["this_month"].append(f)
			else:
				buckets["beyond"].append(f)
		except (ValueError, KeyError):
			# Skip invalid dates
			continue
	
	return buckets


def format_bucket_name(bucket: str) -> str:
	"""Format bucket name for display."""
	return {
		"this_week": "This week",
		"next_2_weeks": "Next 2 weeks",
		"this_month": "This month (days 15-30)",
		"beyond": "Beyond 30 days"
	}.get(bucket, bucket)


def format_date(date_str: str) -> str:
	"""Format ISO date string for display."""
	try:
		dt = parse_iso8601_z(date_str)
		return dt.strftime("%Y-%m-%d")
	except (ValueError, TypeError):
		return date_str


def print_budget_timeline(forecasts: List[Dict], as_of: Optional[datetime] = None) -> None:
	"""Print a timeline view of upcoming expenses grouped by time buckets."""
	if as_of is None:
		as_of = datetime.now(timezone.utc)
	
	# Filter to only active packages with order dates and costs (exclude finished and reordered)
	active_forecasts = [
		f for f in forecasts 
		if not f.get("finished") and not f.get("reordered") and f.get("order_by_date") and f.get("package_cost")
	]
	
	if not active_forecasts:
		print("No upcoming orders with cost information.")
		return
	
	buckets = group_by_time_buckets(active_forecasts, as_of)
	
	print(f"Upcoming expenses (as of {as_of.strftime('%Y-%m-%d')}):")
	print()
	
	total_cost = 0.0
	
	for bucket_name in ["this_week", "next_2_weeks", "this_month", "beyond"]:
		items = buckets[bucket_name]
		if not items:
			continue
		
		# Sort by order_by_date
		items.sort(key=lambda x: x.get("order_by_date", ""))
		
		bucket_total = sum(item["package_cost"] for item in items if item.get("package_cost"))
		total_cost += bucket_total
		
		print(f"{format_bucket_name(bucket_name)}: ${bucket_total:.2f} ({len(items)} order{'s' if len(items) != 1 else ''})")
		for item in items:
			order_date = format_date(item.get("order_by_date", ""))
			cost = item.get("package_cost", 0)
			name = item.get("name", item.get("package_id", "Unknown"))
			print(f"  - {order_date}: ${cost:.2f} ({name})")
		print()
	
	print(f"Total upcoming expenses: ${total_cost:.2f}")


def main():
	"""Main entry point."""
	forecasts = get_all_forecasts()
	print_budget_timeline(forecasts)


if __name__ == "__main__":
	main()
