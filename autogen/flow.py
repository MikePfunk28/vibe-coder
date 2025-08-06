# Add to your flow.py
class EmailAction(Node):
    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        history = shared.get("history", [])
        last_action = history[-1]
        
        return {
            "action_type": last_action["params"].get("action_type", "send"),
            "recipient": last_action["params"].get("recipient", ""),
            "subject": last_action["params"].get("subject", ""),
            "message": last_action["params"].get("message", ""),
            "email_provider": last_action["params"].get("email_provider", "gmail")
        }
    
    def exec(self, params: Dict[str, Any]) -> Dict[str, Any]:
        import asyncio
        from autogen.autogen import EmailBrowserAgent
        
        agent = EmailBrowserAgent(email_provider=params["email_provider"])
        
        if params["action_type"] == "send":
            result = asyncio.run(agent.send_email(
                recipient=params["recipient"],
                subject=params["subject"],
                message=params["message"]
            ))
        elif params["action_type"] == "check_inbox":
            result = asyncio.run(agent.check_inbox())
        
        return result